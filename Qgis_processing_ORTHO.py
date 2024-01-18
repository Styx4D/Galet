# -*- coding: utf-8 -*-

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterFeatureSink,
                       QgsField, QgsFields, QgsFeature,
                       QgsRasterLayer,
                       QgsProject, 
                       QgsCoordinateReferenceSystem,
                       QgsWkbTypes,
                       QgsGeometry,
                       QgsProcessingParameterFile,
                       QgsPointXY)
from qgis import processing

import os, tempfile
import sys
from qgis.PyQt.QtCore import QVariant
from osgeo import gdal, osr
import numpy as np

import rasterio
from rasterio import Affine as A
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from shapely.geometry import box
from rtree import index

class GRANULO_Georef_FINAL(QgsProcessingAlgorithm):
    
    #INPUT
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_WEIGHT = 'INPUT_WEIGHT'
    INPUT_VECTOR = 'INPUT_VECTOR' #vecteur de découpe 
    
    #OUTPUT
    OUTPUT_MASK = 'OUTPUT_MASK' #vecteur polygone des grains identifiés
   
    #config general
    CUT_RAST = 'CUT_RAST' #longueur case
    CUT_SUPERPOS = 'CUT_SUPERPOS' #taux de recouvrement grille
    FILTRE_REC_RESULT = 'FILTRE_REC_RESULT' #tx de recouvrement grain
    
    #config mrcnn
    IMAGE_MAX_DIM = 'IMAGE_MAX_DIM'
    DETECTION_MAX_INSTANCES = 'DETECTION_MAX_INSTANCES'
    RPN_NMS_THRESHOLD = 'RPN_NMS_THRESHOLD'
    POST_NMS_ROIS_INFERENCE = 'POST_NMS_ROIS_INFERENCE'
    DETECTION_NMS_THRESHOLD = 'DETECTION_NMS_THRESHOLD'
    PRE_NMS_LIMIT = 'PRE_NMS_LIMIT'

    
    def __init__(self):
        super().__init__()
        
    def tr(self, string):
               return QCoreApplication.translate('GRANULO_Georef_FINAL', string)

    def createInstance(self):
        return type(self)()

    def name(self):
                return 'GRANULO_Georef_FINAL'

    def displayName(self):
               return self.tr('GRANULO_Georef_FINAL')

    def group(self):
               return self.tr('GRANULO')

    def groupId(self):
              return 'GRANULO'
              
    def shortHelpString(self):
        return self.tr(" /!\ Les données (input et output) seront automatiquement reprojeté sur le crs du projet en cours si aucun crs n'est définie.\nSi le crs du projet est invalide il sera automatiquement definie en EPSG 4326")

    def initAlgorithm(self, config=None):

       
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr('Raster à traiter :')))
                
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_WEIGHT,
                self.tr('Fichier Weight d entrainement (format *.h5):')))
                
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VECTOR,
                self.tr('Polygone de découpe du Raster')))
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.FILTRE_REC_RESULT,
                self.tr('###########################################\nConfig General:\n###########################################\ntaux de recouvrement pour elimination des grains en double'),
                QgsProcessingParameterNumber.Double,
                defaultValue =0.7))
                
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CUT_RAST,
                self.tr('longueur de découpe des cases (pixel)'),
                QgsProcessingParameterNumber.Double,
                defaultValue =512))
                
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CUT_SUPERPOS,
                self.tr('Taux de recouvrement des cases (0 a 1)'),
                QgsProcessingParameterNumber.Double,
                defaultValue =0.1))
                
        self.addParameter(
            QgsProcessingParameterNumber(
                self.IMAGE_MAX_DIM,
                self.tr('###########################################\nConfig MRCNN :\n###########################################\nDimension max de l image /!\ multiple de 256 (256, 512, 1024, 4096...)'),
                QgsProcessingParameterNumber.Integer,
                defaultValue =512))
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DETECTION_MAX_INSTANCES,
                self.tr('Nombre maximum final de détection'),
                QgsProcessingParameterNumber.Integer,
                defaultValue=1000))
                
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RPN_NMS_THRESHOLD,
                self.tr('Seuil de suppression (0 à 1) du RPN'),
                QgsProcessingParameterNumber.Double,
                defaultValue=0.7))
                
        self.addParameter(
            QgsProcessingParameterNumber(
                self.POST_NMS_ROIS_INFERENCE,
                self.tr('Nombre de ROI maximum après filtre RPN'),
                QgsProcessingParameterNumber.Integer,
                defaultValue=2000))
                
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DETECTION_NMS_THRESHOLD,
                self.tr('Seuil de suppression (0 à 1) du NMS'),
                QgsProcessingParameterNumber.Double,
                defaultValue=0.3))
                
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PRE_NMS_LIMIT,
                self.tr('ROI maximum avant filtre NMS'),
                QgsProcessingParameterNumber.Integer,
                defaultValue=9000))
        
            
        self.addParameter(
            QgsProcessingParameterFeatureSink(
            self.OUTPUT_MASK,
            self.tr('###########################################\nContour des grains identifié :')))
        
             
    def processAlgorithm(self, parameters, context, feedback):
            
        ############################################################
        #Initialisation des variables :
        ############################################################
        
        #Temporary directory
        tempdir7 = tempfile.TemporaryDirectory()
        temp_path = tempdir7.name
        
        
        #Image/Raster :
        RAS_IM = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        
        #vecteur de découpe
        CUT = self.parameterAsVectorLayer(parameters, self.INPUT_VECTOR, context)
        
        #weigth file
        input_weight = self.parameterAsFile(parameters,self.INPUT_WEIGHT,context)
        
        #output mask
        outMask = QgsFields()
        outMask.append(QgsField("Len", QVariant.String))
        
        #config general
        Len_case_grille = self.parameterAsDouble(parameters, self.CUT_RAST, context)
        tx_sup_gri = self.parameterAsDouble(parameters, self.CUT_SUPERPOS, context)
        tx_sup_grain = self.parameterAsDouble(parameters, self.FILTRE_REC_RESULT, context)
        
        #projection des couches en cas de CRS invalide
        
        #Project CRS
        user_crs=QgsCoordinateReferenceSystem(QgsProject.instance().crs().authid())
        
        if not user_crs.isValid():
            user_crs = QgsCoordinateReferenceSystem(4326)
        
        
        if RAS_IM.crs() != user_crs:
            RAS_IM.setCrs(user_crs)
            
        if CUT.crs() != user_crs:
                CUT.setCrs(user_crs)
        
        
        (Mask_shp, dest_id) = self.parameterAsSink(parameters, self.OUTPUT_MASK, context, outMask, QgsWkbTypes.MultiPolygon, RAS_IM.crs())

        
        
        #########################
        #### mrcnn config #######
        #########################
        
        path_h5_file = os.path.abspath(input_weight) #ficher h5 as path
        from pathlib import Path
        mrcnn_path = str(Path(path_h5_file).parents[2]) #dossier Mask_RCNN
        
        #mrcnn
        sys.path.append(mrcnn_path) #librairie mrcnn local
        import mrcnn.model as modellib
        #from samples.grain import grain #config model
        import grain #config model
        
        config = grain.CustomConfig() #config dans grain.py
        
        #on change quelques paramètres suivant l'utilisateur
        config_a = self.parameterAsInt(parameters, self.IMAGE_MAX_DIM, context)
        config_b = self.parameterAsInt(parameters, self.DETECTION_MAX_INSTANCES, context)
        config_c = self.parameterAsDouble(parameters, self.RPN_NMS_THRESHOLD, context)
        config_d = self.parameterAsInt(parameters, self.POST_NMS_ROIS_INFERENCE, context)
        config_e = self.parameterAsDouble(parameters, self.DETECTION_NMS_THRESHOLD, context)
        config_f = self.parameterAsInt(parameters, self.PRE_NMS_LIMIT, context)
        
        class InferenceConfig(config.__class__):
                IMAGE_MAX_DIM = config_a #default 1024
                DETECTION_MAX_INSTANCES = config_b  #Max number of final detections default 100
                RPN_NMS_THRESHOLD = config_c # Non-max suppression threshold to filter RPN proposals. default 0.7
                POST_NMS_ROIS_INFERENCE = config_d # ROIs kept after non-maximum suppression (training and inference) default 1000
                DETECTION_NMS_THRESHOLD = config_e # Non-maximum suppression threshold for detection default 0.3
                PRE_NMS_LIMIT = config_f # ROIs kept after tf.nn.top_k and before non-maximum suppression default 6000
                DETECTION_MIN_CONFIDENCE = 0.001
                IMAGE_RESIZE_MODE = "square"
                #RPN_ANCHOR_SCALES = (8,16, 32, 64, 128)
        
        config = InferenceConfig()
        
        
        class_names = ['BG', 'grain']
        model_dir = os.path.join(mrcnn_path, "logs")
        
        #chargement du modele
        feedback.pushInfo("Chargement du modéle...")
        
        from keras.backend import clear_session
        clear_session()
        model = modellib.MaskRCNN(mode="inference", model_dir = model_dir, config=config)
        #model.load_weights(path_h5_file, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        model.load_weights(path_h5_file, by_name=True)
        model.keras_model._make_predict_function()
        
        
        
        
        #total des polys a nettoyé si plusieurs cut
        TOTAL_polys_toclean=[]
        #iteration sur cut
        for ite_1, cut_features in enumerate(CUT.getFeatures()):
            
            rast_pxlX = RAS_IM.rasterUnitsPerPixelX()
            rast_pxlY = RAS_IM.rasterUnitsPerPixelY()
            
            len_case_geoX = Len_case_grille * rast_pxlX
            len_case_geoY = Len_case_grille * rast_pxlY
            
            cut_geom_bb = cut_features.geometry().boundingBox()

            #cut the raster on bbox
            
            n_path = temp_path +'\\cut_'+str(ite_1)+'.tif'
            #<x_min> <y_min> <x_max> <y_max>
            bb_gdal= cut_geom_bb.buffered(max(len_case_geoX,len_case_geoY))
            processing.run('gdal:cliprasterbyextent',{
                'INPUT':RAS_IM,
                'PROJWIN':bb_gdal,
                'NODATA':None,
                'OPTIONS':'',
                'DATA_TYPE':0,
                'EXTRA':'',
                'OUTPUT':n_path})
            
            ds = gdal.Open(n_path)

            # stack bands as np array
            band_1 = np.array(ds.GetRasterBand(1).ReadAsArray())
            band_2 = np.array(ds.GetRasterBand(2).ReadAsArray())
            band_3 = np.array(ds.GetRasterBand(3).ReadAsArray())
            
            bands = np.stack((band_1,band_2,band_3),2)

            #free memory
            ds= None
            band_1 = None
            band_2 = None
            band_3 = None

            #get metadata
            cut_rast = QgsRasterLayer(n_path)

            rast_ext = cut_rast.extent()
            
            
            #creation d'une grille de x par x pixel
            feedback.pushInfo("creation de la grille sur la zone n° " + str(ite_1+1))
            feedback.pushInfo("meshing...")
            
            
            cut_width = cut_geom_bb.width()
            cut_height = cut_geom_bb.height()
            
            
            #top left corner
            x_tlc = cut_geom_bb.xMinimum()
            y_tlc = cut_geom_bb.yMaximum()
            
            #incrémente pour avoir une grille clean sur la bbox du feature
            incr_x = (1-tx_sup_gri)*len_case_geoX - cut_width%((1-tx_sup_gri)*len_case_geoX)
            incr_y = (1-tx_sup_gri)*len_case_geoY - cut_height%((1-tx_sup_gri)*len_case_geoY)
            
            #nombre de grille à créer
            ngridx = int((cut_width + incr_x)/((1-tx_sup_gri)*len_case_geoX))
            ngridy = int((cut_height + incr_y)/((1-tx_sup_gri)*len_case_geoY))
            
            #top left avec incr
            x0 = x_tlc - 0.5*incr_x
            y0 = y_tlc + 0.5*incr_y

            points_x = [x0]
            points_y = [y0]
            
            #coord x et y
            #si overlap est nul, une seul iter
            ov_iter =1
            if tx_sup_gri>0:
                ov_iter=2
                
            for i in range(int(ngridx)*ov_iter-1):
                if len(points_x)%2>0 :
                    points_x.append(points_x[i]+len_case_geoX)
                else :
                    points_x.append(points_x[i]-tx_sup_gri*len_case_geoX)
               

            for i in range(int(ngridy)*ov_iter-1):
                if len(points_y)%2>0 :
                    points_y.append(points_y[i]-len_case_geoY)
                else:
                    points_y.append(points_y[i]+tx_sup_gri*len_case_geoY)
                
                        
            #mesh&flat
            gridx, gridy = np.meshgrid(points_x,points_y)
            gridx=gridx.flatten()
            gridy=gridy.flatten()
            
            #converting grid to pxl value
            #coord du raster top left corner
            x_rast_ext = rast_ext.xMinimum()
            y_rast_ext = rast_ext.yMaximum()
            
            #coord en pxl
            feedback.pushInfo("converting grid to pxl...")
            x_pxl_tlc = round((x0-x_rast_ext)/rast_pxlX)
            grid_x_pxl = [round(x_pxl_tlc+(gridx[i]-x0)/rast_pxlX) for i in range(len(gridx))]
            
            y_pxl_tlc = round((y_rast_ext-y0)/rast_pxlY)
            grid_y_pxl = [round(y_pxl_tlc+(y0-gridy[i])/rast_pxlY) for i in range(len(gridy))]
            
            #converting to QgsGeometry
            feedback.pushInfo("conversion des points en Geom")
            
            
            data= [[[i],[],[],[],[],[]] for i in range(int(ngridx*ngridy))] #img, bbox, bbox unoverlap, increment, bbox en pxl, extent = xmini et y maxi
            nimg=0
            nite=1
            
            for a in range(0,len(gridy),int(ngridx)*ov_iter):
                for i in range(0+a,int(ngridx)*ov_iter+a,2):
                    
                    
                    if nite==1:
                        data[nimg][1].extend([QgsPointXY(gridx[i],gridy[i]),QgsPointXY(gridx[i+1],gridy[i+1])])
                        data[nimg][2].extend([QgsPointXY(gridx[i]+tx_sup_gri*len_case_geoX,gridy[i]-tx_sup_gri*len_case_geoY),\
                        QgsPointXY(gridx[i+1]-tx_sup_gri*len_case_geoX,gridy[i+1]-tx_sup_gri*len_case_geoY)])
                        data[nimg][3].extend([i,i+1])
                        data[nimg][4].extend([grid_x_pxl[i],grid_x_pxl[i+1],grid_y_pxl[i]])
                        data[nimg][5].extend([gridx[i],gridy[i]])
                    else:
                        data[nimg][1].extend([QgsPointXY(gridx[i+1],gridy[i+1]),QgsPointXY(gridx[i],gridy[i])])
                        data[nimg][2].extend([QgsPointXY(gridx[i+1]-tx_sup_gri*len_case_geoX,gridy[i+1]+tx_sup_gri*len_case_geoY),\
                        QgsPointXY(gridx[i]+tx_sup_gri*len_case_geoX,gridy[i]+tx_sup_gri*len_case_geoY)])
                        data[nimg][3].extend([i,i+1])
                        data[nimg][4].extend([grid_y_pxl[i+1]])
                    
                    nimg+=1
                
                if nite==1:
                    nimg-=int(ngridx)
                    nite=0
                else:
                    nite=1


            #suppresion des couches qui ne superposent ap'
            feedback.pushInfo("removing non overlaping grid")
            cut_geom = cut_features.geometry()
           
            keep_i = []
            
            for i in range(0,len(data)):
                
                gr_poly = QgsGeometry.fromPolygonXY([data[i][1]])
              
                if gr_poly.intersects(cut_geom):
                    keep_i.append(i)
            
            clean_data = [data[i] for i in keep_i]
            
            
            #detecting 
            results=[]
            feedback.pushInfo("MaskR CNN detection...")
            
            for i in range(len(clean_data)):
                array_d = bands[clean_data[i][4][2]:clean_data[i][4][3],\
                    clean_data[i][4][0]:clean_data[i][4][1],:]
                
                feedback.pushInfo("image "+str(i+1)+" on " +str(len(clean_data))+". Shape: "+str(array_d.shape))
                results.append(model.detect([array_d], verbose=0))
            
            #mask --> SHP
            feedback.pushInfo("converting to vector")
            poly=[]
            id_data=0
            for result in results:
                for r in result:
                    for i in range(r['rois'].shape[0]):
                        
                        mask_array = np.array(r['masks'][:, :, i],dtype=np.uint8)
                        #rasterio : (array, masque (non digitalisation de 0), transformation)
                        for vec in rasterio.features.shapes(mask_array, mask_array,
                            transform=A.translation(clean_data[id_data][5][0],
                            clean_data[id_data][5][1]) * A.scale(rast_pxlX, -rast_pxlY)):
                            
                            poly.append(vec)
                id_data+=1
            res = ({'geometry': s} for i, (s, _) in enumerate(poly))
            
            geoms = list(res)
            feedback.pushInfo("converting to QgsGeometry() type")
            geoms_coords =[geoms[i]["geometry"]["coordinates"][0] for i in range(len(geoms))]
            
            polys = [QgsGeometry.fromPolygonXY([[QgsPointXY(geoms_coords[i][j][0],geoms_coords[i][j][1]) \
                                for j in range(len(geoms_coords[i]))]]) for i in range(len(geoms_coords)) ]
            
            polys_bb = [pp.boundingBox() for pp in polys]
            #cleaning polygons
            feedback.pushInfo("cleaning "+str(len(polys))+" polygons")

            feedback.pushInfo("Creating spatial index...")
            
            #init Rtree
            idx_rtree = index.Index()
            
            #convert to shapely box
            box_shply = [box(u.boundingBox().xMinimum(),u.boundingBox().yMinimum(),\
            u.boundingBox().xMaximum(),u.boundingBox().yMaximum()) for u in polys]

            # Populate R-tree index with bounds of grid cells
            for pos, cell in enumerate(box_shply):
                idx_rtree.insert(pos, cell.bounds)

            # Loop through each Shapely polygon
            list_comp_id=[]
            for poly in box_shply:
                a=list(idx_rtree.intersection(poly.bounds))
                list_comp_id.append(a)
            
            feedback.pushInfo("computing IoU")
            to_merge=[[id,idd] for id in range(len(list_comp_id)) for idd in list_comp_id[id]\
                if (len(list_comp_id[id])>1 and (idd!=id) and (\
                    polys_bb[id].intersect(polys_bb[idd]).area()/polys_bb[id].area()>tx_sup_grain or \
                    polys_bb[id].intersect(polys_bb[idd]).area()/polys_bb[idd].area()>tx_sup_grain))]
            
            #sort and set
            mer_so = [sorted(sub) for sub in to_merge]
            mer_se = [list(item) for item in set(tuple(row) for row in mer_so)]
            
            #merging polygons
            feedback.pushInfo("merging  les doublons")
            to_del=[]
            for id in mer_se:
                polys[id[0]]=QgsGeometry.unaryUnion([polys[id[0]],polys[id[1]]])
                to_del.append(id[1])
                
            polygon_cleany = [polys[i] for i in range(len(polys)) if i not in to_del]
            
            
            
            
            if CUT.featureCount() >1:
                feedback.pushInfo("adding result..")
                TOTAL_polys_toclean.extend(polygon_cleany)
            else:
                feedback.pushInfo("writing file...")
                for poly in polygon_cleany:
                    f = QgsFeature()
                    f.setGeometry(poly)
                    dist = 2*np.sqrt(poly.closestSegmentWithContext(poly.centroid().asPoint())[0])
                    if dist > 2*rast_pxlX:
                        f.setAttributes([str(dist)])
                        Mask_shp.addFeature(f , QgsFeatureSink.FastInsert)
            
        #si la zone est fractionné nettoie tout ça
        if CUT.featureCount() >1:
            #!!! le nettoyage se fait sur bbox : faire des traits droit :3
            #buff les zones sur 20 pxl
            
            feedback.pushInfo("multiple zone provided")
            feedback.pushInfo("calculating buffer")
            list_buff = []
            for cut_feat in CUT.getFeatures():
                list_buff.append(cut_feat.geometry().buffer(250*rast_pxlX,4))
            
            #calculating crossed buffer
            buff_cross_id = [[a,b] for a in range(len(list_buff)) for b in range(len(list_buff)) \
                if list_buff[a].intersects(list_buff[b]) and not list_buff[a].equals(list_buff[b])]
            
            #set
            buff_so = [sorted(sub) for sub in buff_cross_id]
            buff_se = [list(item) for item in set(tuple(row) for row in buff_so)]

            #to geom
            buff_cross= [list_buff[a[0]].intersection(list_buff[a[1]]) for a in buff_se]
            
            #idx
            feedback.pushInfo("creation d'un index spatial")
             #init Rtree
            idx_rtree_bis = index.Index()
            
            #convert grain to shapely box
            box_shply_total = [box(u.boundingBox().xMinimum(),u.boundingBox().yMinimum(),\
            u.boundingBox().xMaximum(),u.boundingBox().yMaximum()) for u in TOTAL_polys_toclean]
            
            #convert buffer to shapely box
            box_shply_buff = [box(u.boundingBox().xMinimum(),u.boundingBox().yMinimum(),\
                u.boundingBox().xMaximum(),u.boundingBox().yMaximum()) for u in buff_cross]
            
            # Populate R-tree index 
            for pos, cell in enumerate(box_shply_total):
                idx_rtree_bis.insert(pos, cell.bounds)

            # Loop through each 
            list_comp_id=[]
            for poly in box_shply_buff:
                #a -> liste des grains dans un buffer
                a=list(idx_rtree_bis.intersection(poly.bounds))
                list_comp_id.extend(a)
                
            #loop sur les grains des buffers
            shpy_on_buffer=[box_shply_total[id_li] for id_li in list_comp_id]
                
            list_comp_id_bis=[]
            for poly in shpy_on_buffer:
                a=list(idx_rtree_bis.intersection(poly.bounds))
                list_comp_id_bis.append(a)
            
            feedback.pushInfo("computing IoU")
            
            to_merge=[[id_1,id_2] for id in range(len(list_comp_id_bis)) for id_1 in list_comp_id_bis[id] for id_2 in list_comp_id_bis[id]\
                if (len(list_comp_id_bis[id])>1 and id_1!=id_2 and (\
                    TOTAL_polys_toclean[id_1].boundingBox().intersect(TOTAL_polys_toclean[id_2].boundingBox()).area()\
                    /TOTAL_polys_toclean[id_1].boundingBox().area()>tx_sup_grain or \
                    TOTAL_polys_toclean[id_1].boundingBox().intersect(TOTAL_polys_toclean[id_2].boundingBox()).area()\
                    /TOTAL_polys_toclean[id_2].boundingBox().area()>tx_sup_grain))]
            
            #sort and set
            mer_so = [sorted(sub) for sub in to_merge]
            mer_se = [list(item) for item in set(tuple(row) for row in mer_so)]
            
    
            #merging polygons
            feedback.pushInfo("merging  les doublons")
            to_del=[]
            for id in mer_se:
                TOTAL_polys_toclean[id[0]]=QgsGeometry.unaryUnion([TOTAL_polys_toclean[id[0]],TOTAL_polys_toclean[id[1]]])
                to_del.append(id[1])
                
            polygon_cleany = [TOTAL_polys_toclean[i] for i in range(len(TOTAL_polys_toclean)) if i not in to_del]
            
            
            feedback.pushInfo("writing file...")
        
            for poly in polygon_cleany:
                f = QgsFeature()
                f.setGeometry(poly)
                dist = 2*np.sqrt(poly.closestSegmentWithContext(poly.centroid().asPoint())[0])
                if dist > 2*rast_pxlX:
                    f.setAttributes([str(dist)])
                    Mask_shp.addFeature(f , QgsFeatureSink.FastInsert)
                    
        return{}

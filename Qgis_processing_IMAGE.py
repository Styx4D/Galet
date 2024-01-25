# -*- coding: utf-8 -*-

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsFeatureSink,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterFeatureSink,
                       QgsField, QgsFields, QgsFeature,
                       QgsRasterLayer,
                       QgsWkbTypes,
                       QgsGeometry,
                       QgsProcessingParameterFile,
                       QgsPointXY)
from qgis import processing

import os, tempfile
import sys
from qgis.PyQt.QtCore import QVariant
from osgeo import gdal
import numpy as np

import rasterio
from rasterio import Affine as A
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

from PIL import Image
from pathlib import Path

from keras.backend import clear_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

class GALET_image(QgsProcessingAlgorithm):
    
    #INPUT
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_WEIGHT = 'INPUT_WEIGHT'
    
    SCALE_LINE = 'SCALE_LINE'
    SCALE_LEN = 'SCALE_LEN'
    
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
    OUTPUT_RAST = 'OUTPUT_RAST'
    
    def __init__(self):
        super().__init__()
        
    def tr(self, string):
               return QCoreApplication.translate('GALET_image', string)

    def createInstance(self):
        return type(self)()

    def name(self):
                return 'GALET_image'

    def displayName(self):
               return self.tr('GALET_image')

    def group(self):
               return self.tr('Galet')

    def groupId(self):
              return 'Galet'
              
    def shortHelpString(self):
        return self.tr(" Qgis Processing for pebbles detection on image")

    def initAlgorithm(self, config=None):

       
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr('Input Image :')))
                
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_WEIGHT,
                self.tr('Weight file (format *.h5):')))
                
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.SCALE_LINE,
                self.tr('Scale line:')))
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SCALE_LEN,
                self.tr('Scale Size (meters)'),
                QgsProcessingParameterNumber.Double,
                defaultValue =10))
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.FILTRE_REC_RESULT,
                self.tr('###########################################\nGeneral Settings:\n###########################################\nOverlap rate for removing duplicate grains'),
                QgsProcessingParameterNumber.Double,
                defaultValue =0.8))
                
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CUT_RAST,
                self.tr('Cut length of the cells (pixels)'),
                QgsProcessingParameterNumber.Double,
                defaultValue =512))
                
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CUT_SUPERPOS,
                self.tr('Overlap rate of the cells (0 to 1)'),
                QgsProcessingParameterNumber.Double,
                defaultValue =0.1))
                
              
        self.addParameter(
            QgsProcessingParameterNumber(
                self.IMAGE_MAX_DIM,
                self.tr('###########################################\nMRCNN Settings:\n###########################################\nMaximum image dimension /!\ multiple of 256 (256, 512, 1024, 4096...)'),
                QgsProcessingParameterNumber.Integer,
                defaultValue =512))
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DETECTION_MAX_INSTANCES,
                self.tr('Maximum final detection count per cell'),
                QgsProcessingParameterNumber.Integer,
                defaultValue=1000))
                
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RPN_NMS_THRESHOLD,
                self.tr('Suppression threshold (0 to 1) for RPN'),
                QgsProcessingParameterNumber.Double,
                defaultValue=0.7))
                
        self.addParameter(
            QgsProcessingParameterNumber(
                self.POST_NMS_ROIS_INFERENCE,
                self.tr('Maximum number of ROI after RPN filter'),
                QgsProcessingParameterNumber.Integer,
                defaultValue=2000))
                
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DETECTION_NMS_THRESHOLD,
                self.tr('Suppression threshold (0 to 1) for NMS'),
                QgsProcessingParameterNumber.Double,
                defaultValue=0.3))
                
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PRE_NMS_LIMIT,
                self.tr('Maximum ROI before NMS filter.'),
                QgsProcessingParameterNumber.Integer,
                defaultValue=9000))
        
            
        self.addParameter(
            QgsProcessingParameterFeatureSink(
            self.OUTPUT_MASK,
            self.tr('###########################################\nContour of identified grains')))
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
            self.OUTPUT_RAST,
            self.tr('Scaled Image :')))
        
        
             
    def processAlgorithm(self, parameters, context, feedback):
            
        ############################################################
        #Initialisation des variables :
        ############################################################
        
        #Temporary directory
        tempdir7 = tempfile.TemporaryDirectory()
        temp_path = tempdir7.name
        
        
        #Image/Raster :
        RAS_IM = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        OUT_RAS = self.parameterAsOutputLayer(parameters, self.OUTPUT_RAST, context)
        
        #scale
        SCALE_LEN = self.parameterAsDouble(parameters, self.SCALE_LEN, context)
        SCALE_LINE = self.parameterAsVectorLayer(parameters, self.SCALE_LINE, context)
        
        #weigth file
        input_weight = self.parameterAsFile(parameters,self.INPUT_WEIGHT,context)
        
        #output mask
        outMask = QgsFields()
        outMask.append(QgsField("Len", QVariant.String))
        
        #config general
        Len_case_grille = self.parameterAsDouble(parameters, self.CUT_RAST, context)
        tx_sup_gri = self.parameterAsDouble(parameters, self.CUT_SUPERPOS, context)
        tx_sup_grain = self.parameterAsDouble(parameters, self.FILTRE_REC_RESULT, context)
        
        #georeferencement de l'image
        
        im_file = RAS_IM.source()
        false_len = SCALE_LINE.getFeature(1).geometry().length()
        scale =  SCALE_LEN /false_len
        
        im = Image.open(im_file)
        im.save(OUT_RAS, 'TIFF')

        gdal.AllRegister()
        rast_src = gdal.Open(OUT_RAS, 1 )

        gt = rast_src.GetGeoTransform()
        gtl = list(gt)
        gtl[0] = 20000
        _, ytodel = im.size
        gtl[3] = 20000-ytodel
        gtl[1] =  -scale
        gtl[5] =  scale
        rast_src.SetGeoTransform(tuple(gtl))
        rast_src = None
        
        (Mask_shp, dest_id) = self.parameterAsSink(parameters, self.OUTPUT_MASK, context, outMask, QgsWkbTypes.MultiPolygon, RAS_IM.crs())

        
        
        #########################
        #### mrcnn config #######
        #########################
        
        path_h5_file = os.path.abspath(input_weight) #ficher h5 as path
        
        mrcnn_path = str(Path(path_h5_file).parents[2]) #dossier Mask_RCNN
        
        #mrcnn
        sys.path.append(mrcnn_path) #librairie mrcnn local
        import mrcnn.model as modellib
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
        
        model_dir = os.path.join(mrcnn_path, "logs")
        
        #chargement du modele
        feedback.pushInfo("Model loading...")
        
        clear_session()
        model = modellib.MaskRCNN(mode="inference", model_dir = model_dir, config=config)

        model.load_weights(path_h5_file, by_name=True)
        model.keras_model._make_predict_function()
        
        #loading raster 
        #open the raster in gdal
        RAS_IM_gdal = gdal.Open(OUT_RAS)
        
        band_1 = np.array(RAS_IM_gdal.GetRasterBand(1).ReadAsArray())
        band_2 = np.array(RAS_IM_gdal.GetRasterBand(2).ReadAsArray())
        band_3 = np.array(RAS_IM_gdal.GetRasterBand(3).ReadAsArray())
        
        bands = np.stack((band_1,band_2,band_3),2)
        
        OUT_RAS=QgsRasterLayer(OUT_RAS)
        rast_ext = OUT_RAS.extent()
        
        rast_pxlX = OUT_RAS.rasterUnitsPerPixelX()
        rast_pxlY = OUT_RAS.rasterUnitsPerPixelY()
        
        len_case_geoX = Len_case_grille * rast_pxlX
        len_case_geoY = Len_case_grille * rast_pxlY
        
        #creation d'une grille de x par x pixel
        feedback.pushInfo("meshing...")
        
        cut_geom_bb = rast_ext
        cut_width = cut_geom_bb.width()
        cut_height = cut_geom_bb.height()
        
        #top left corner
        x_tlc = cut_geom_bb.xMinimum()
        y_tlc = cut_geom_bb.yMaximum()
        
        #nombre de grille à créer
        ngridx = int((cut_width)/((1-tx_sup_gri)*len_case_geoX))
        ngridy = int((cut_height)/((1-tx_sup_gri)*len_case_geoY))
        
        points_x = [x_tlc]
        points_y = [y_tlc]
        
        x0=x_tlc
        y0=y_tlc
        
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
            
        #last one pour tout recouvrir
        points_x.extend([x_tlc+cut_width-len_case_geoX,x_tlc+cut_width])
        points_y.extend([y_tlc-cut_height+len_case_geoY,y_tlc-cut_width])
        
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
        feedback.pushInfo("Converting to QgsGeometry")
        
        data= [[[i],[],[],[],[],[]] for i in range(int((ngridx+1)*(ngridy+1)))] # img, bbox, bbox unoverlap, increment, bbox pxl, extent = x_mini & y_maxi
        nimg=0
        nite=1

        for a in range(0,len(gridy),2+int(ngridx)*ov_iter):
            for i in range(0+a,2+int(ngridx)*ov_iter+a,2):
                
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
                nimg-=int(ngridx)+1
                nite=0
            else:
                nite=1

        #detecting 
        results=[]
        feedback.pushInfo("MaskR CNN detection...")
        
        for i in range(len(data)):
            array_d = bands[ data[i][4][2]:data[i][4][3], data[i][4][0]:data[i][4][1], :]

            print("image ",i+1," on " ,len(data),". Shape: ",array_d.shape)
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
                        transform=A.translation(data[id_data][5][0],
                        data[id_data][5][1]) * A.scale(rast_pxlX, -rast_pxlY)):
                        
                        poly.append(vec)
            id_data+=1
        res = ({'geometry': s} for i, (s, _) in enumerate(poly))
        
        geoms = list(res)
        feedback.pushInfo("converting to QgsGeometry() type")
        geoms_coords =[geoms[i]["geometry"]["coordinates"][0] for i in range(len(geoms))]
        
        polygon = [QgsGeometry.fromPolygonXY([[QgsPointXY(geoms_coords[i][j][0],geoms_coords[i][j][1]) \
                            for j in range(len(geoms_coords[i]))]]) for i in range(len(geoms_coords)) ]
        
        #cleaning polygons
        feedback.pushInfo("cleaning "+str(len(polygon))+" polygons")
        feedback.pushInfo("remove doublons")
        center_p = []
        double_pol = []
        for i in range(len(polygon)):
            poly = polygon[i]
            if [poly.centroid().asPoint().x(),poly.centroid().asPoint().y()] not in center_p :
                center_p.append([poly.centroid().asPoint().x(),poly.centroid().asPoint().y()])
            else:
                double_pol.append(i)
        
        polygon = [polygon[i] for i in range(len(polygon)) if i not in double_pol]

        #calculation clean area = grain on bbox unverlap
        
        unover_poly = [QgsGeometry.fromPolygonXY([data[i][2]]) for i in range(len(data))]
        unover_merge = QgsGeometry.unaryUnion(unover_poly)
        
        over_poly = [QgsGeometry.fromPolygonXY([data[i][1]]) for i in range(len(data))]
        over_merge = QgsGeometry.unaryUnion(over_poly)
        
        over_unover = over_merge.difference(unover_merge)
        
        feedback.pushInfo("determination des poly to clean")
        #polygon_to_clean = [polygon[i] for i in range(len(polygon)) if polygon[i].intersects(over_unover)]
        polygon_to_clean = [polygon[i] for i in range(len(polygon))]
        feedback.pushInfo("remaining..")
        #polygon_clean = [polygon[i] for i in range(len(polygon)) if not polygon[i].intersects(over_unover)]
        
        feedback.pushInfo("calculating crossed...")
        id_inter = [[[i,j] for i in range(len(polygon_to_clean)) if polygon_to_clean[i].boundingBoxIntersects(polygon_to_clean[j]) and i!=j]\
                            for j in range(len(polygon_to_clean)) ]
        #flat it
        id_inter = [item for sublist in id_inter for item in sublist]
        #sort to avoid doublon
        sort_id = [sorted(id_inter[i]) for i in range(len(id_inter)) if any(id_inter[i])]
        #set it
        sort_id = list(set(tuple(i) for i in sort_id))
         
        feedback.pushInfo("computing IoU")
        to_merge =[sort_id[i][:] for i in range(len(sort_id)) if \
            polygon_to_clean[sort_id[i][0]].intersection(polygon_to_clean[sort_id[i][1]]).area()\
            /polygon_to_clean[sort_id[i][1]].area()>0.7 or \
            polygon_to_clean[sort_id[i][0]].intersection(polygon_to_clean[sort_id[i][1]]).area()\
            /polygon_to_clean[sort_id[i][0]].area()>0.7 ]
        
        #merging polygons
        feedback.pushInfo("merging  les doublons")
        to_del=[]
        for id in to_merge:
            polygon_to_clean[id[0]]=QgsGeometry.unaryUnion([polygon_to_clean[id[0]],polygon_to_clean[id[1]]])
            to_del.append(id[1])
            
        polygon_cleany = [polygon_to_clean[i] for i in range(len(polygon_to_clean)) if i not in to_del]
        
        
        
        feedback.pushInfo("writing file...")
        """
        if any(polygon_clean):
            for poly in polygon_clean:
                f = QgsFeature()
                f.setGeometry(poly)
                #axe b
                dist = 2*np.sqrt(poly.closestSegmentWithContext(poly.centroid().asPoint())[0])
                f.setAttributes([str(dist)])
                Mask_shp.addFeature(f , QgsFeatureSink.FastInsert)
        """
        if any(polygon_cleany):
            for poly in polygon_cleany:
                f = QgsFeature()
                f.setGeometry(poly)
                dist = 2*np.sqrt(poly.closestSegmentWithContext(poly.centroid().asPoint())[0])
                f.setAttributes([str(dist)])
                Mask_shp.addFeature(f , QgsFeatureSink.FastInsert)
            
        
        return{}

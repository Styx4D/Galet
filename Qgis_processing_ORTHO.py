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

import subprocess
try: 
    import rasterio
    from rasterio import Affine as A
    from rasterio import features as rasterio_features
except:
    subprocess.check_call(['python', '-m', 'pip', 'install', 'rasterio==1.3.9'])
    import rasterio
    from rasterio import Affine as A
    from rasterio import features as rasterio_features

try : 
    from shapely.geometry import box
except:
    subprocess.check_call(['python', '-m', 'pip', 'install', 'shapely'])
    from shapely.geometry import box

try : 
    from rtree import index
except:
    subprocess.check_call(['python', '-m', 'pip', 'install', 'rtree'])
    from rtree import index

try:
    import cv2 as cv
except:
    subprocess.check_call(['python', '-m', 'pip', 'install', 'opencv-python==4.9.0'])
    import cv2 as cv

from pathlib import Path

import socket
import pickle

import copy
from itertools import chain

class GALET_Georef(QgsProcessingAlgorithm):
    
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

    HOST, PORT = "localhost", 9999
    BUFF_SIZE = 65536

    def __init__(self):
        super().__init__()
        
    def tr(self, string):
               return QCoreApplication.translate('GALET_Georef', string)

    def createInstance(self):
        return type(self)()

    def name(self):
                return 'GALET_Georef'

    def displayName(self):
               return self.tr('GALET_Georef')

    def group(self):
               return self.tr('Galet')

    def groupId(self):
              return 'Galet'
              
    def shortHelpString(self):
        return self.tr(" /!\ Les données (input et output) seront automatiquement reprojeté sur le crs du projet en cours si aucun crs n'est définie.\nSi le crs du projet est invalide il sera automatiquement definie en EPSG 4326")

    def initAlgorithm(self, config=None):

       
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr('Input Raster :')))
                
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_WEIGHT,
                self.tr('Weight file (format *.h5):')))
                
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VECTOR,
                self.tr('Raster Clipping Polygon')))
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.FILTRE_REC_RESULT,
                self.tr('###########################################\nGeneral Settings:\n###########################################\nOverlap rate for removing duplicate grains'),
                QgsProcessingParameterNumber.Double,
                defaultValue =0.7))
                
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
        

    def exchange_with_server(self, instructions_dict):
        # send/receive data from conda env

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.HOST, self.PORT))
            #pickled_dict = pickle.dumps()

            print("sending data...")
            sock.sendall(pickle.dumps(instructions_dict))
            print("... data sent")

            data = []
            while True:
                try :
                    packet = sock.recv(self.BUFF_SIZE)
                    if len(packet)<self.BUFF_SIZE:
                        data.append(packet)
                        break
                    elif not packet:
                        break
                except : 
                    break
                data.append(packet)
                time.sleep(0.001)
                # set it only after the 1st receive so that it wan wait for the computation to happen
                sock.settimeout(1.)
                  
            received_response = pickle.loads(b"".join(data))

            return received_response
        return {'status':'Connection failed'}

    def call_inference(self, img, SA):
        imshared, shm_img, transfer_dict = SA.shareNPArray(img)

        # Send it to the server and wait for the response
        response_dict = self.exchange_with_server({'infer':transfer_dict})

        shared_msks, shm_msks = SA.readSharedNPArray(response_dict['masks'])
        shared_rois, shm_rois = SA.readSharedNPArray(response_dict['rois'])
        shared_cids, shm_cids = SA.readSharedNPArray(response_dict['class_ids'])
        shared_scrs, shm_scrs = SA.readSharedNPArray(response_dict['scores'])

        # put them in the correct data structure
        result = [{'masks':copy.deepcopy(shared_msks), 'rois':copy.deepcopy(shared_rois), 'class_ids':copy.deepcopy(shared_cids), 'scores':copy.deepcopy(shared_scrs)}]

        # release the shared memory
        SA.closeSharedNPArrays([shm_msks, shm_rois, shm_cids, shm_scrs, shm_img], unlink=True)

        return result


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

        mrcnn_path = str(Path(path_h5_file).parents[2]) #dossier Mask_RCNN
        
        #mrcnn
        sys.path.append(mrcnn_path) #librairie mrcnn local
        import ShareArrays as SA

        max_dim_infer = self.parameterAsInt(parameters, self.IMAGE_MAX_DIM, context)
        config_dict = {
            'config': {
                self.IMAGE_MAX_DIM           : max_dim_infer,
                self.DETECTION_MAX_INSTANCES : self.parameterAsInt(parameters, self.DETECTION_MAX_INSTANCES, context),
                self.RPN_NMS_THRESHOLD       : self.parameterAsDouble(parameters, self.RPN_NMS_THRESHOLD, context),
                self.POST_NMS_ROIS_INFERENCE : self.parameterAsInt(parameters, self.POST_NMS_ROIS_INFERENCE, context),
                self.DETECTION_NMS_THRESHOLD : self.parameterAsDouble(parameters, self.DETECTION_NMS_THRESHOLD, context),
                self.PRE_NMS_LIMIT           : self.parameterAsInt(parameters, self.PRE_NMS_LIMIT, context),
                'path_h5_file'          : path_h5_file
                }
            }
        
        #chargement du modele
        feedback.pushInfo("Model loading...")
        ret_dict = self.exchange_with_server(config_dict)
        feedback.pushInfo(ret_dict['status'])

        
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
            feedback.pushInfo("Creating grid n° " + str(ite_1+1))
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
            feedback.pushInfo("Removing non overlaping grid")
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
                
                prev_shape = array_d.shape[:2]
                array_d = cv.resize( array_d, (max_dim_infer, max_dim_infer))
            
                feedback.pushInfo("image "+str(i+1)+" on " +str(len(clean_data))+". Shape: "+str(array_d.shape))
                r = self.call_inference(array_d, SA)
            
                mask_array = np.array(r[0]['masks'],dtype=np.uint8)
                r[0]['masks'] = cv.resize( mask_array, prev_shape)
                results.append(r)
            
            #mask --> SHP
            feedback.pushInfo("converting to vector")
            poly=[]
            id_data=0
            for result in results:
                for r in result:
                    for i in range(r['rois'].shape[0]):
                        
                        mask_array = np.array(r['masks'][:, :, i],dtype=np.uint8)
                        
                        #rasterio : (array, masque (non digitalisation de 0), transformation)
                        for vec in rasterio_features.shapes(mask_array, mask_array,
                            transform=A.translation(clean_data[id_data][5][0],
                            clean_data[id_data][5][1]) * A.scale(rast_pxlX, -rast_pxlY)):
                            
                            poly.append(vec)
                id_data+=1
            
            res = ({'geometry': s} for i, (s, _) in enumerate(poly))
            feedback.pushInfo(str(len(poly)))
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
            feedback.pushInfo("merging duplicates")

            class UnionFind:
                def __init__(self):
                    self.parent = {}

                def find(self, u):
                    if u != self.parent.setdefault(u, u):
                        self.parent[u] = self.find(self.parent[u])
                    return self.parent[u]

                def union(self, u, v):
                    root_u, root_v = self.find(u), self.find(v)
                    if root_u != root_v:
                        self.parent[root_u] = root_v

            def merge_lists_with_common_ids(id_list):
                union_find = UnionFind()

                for ids in id_list:
                    for id_ in ids:
                        union_find.find(id_)

                for ids in id_list:
                    union_find.union(ids[0], ids[1])

                merged_lists = {}
                for id_ in union_find.parent.keys():
                    root = union_find.find(id_)
                    if root not in merged_lists:
                        merged_lists[root] = [id_]
                    else:
                        merged_lists[root].append(id_)

                return list(merged_lists.values())

            to_del= list(chain.from_iterable(mer_se))
            to_merge_comb = merge_lists_with_common_ids(to_merge)
            to_ext = []
            for list_comb_ids in to_merge_comb:
                list_comb = [ polys[i] for i in list_comb_ids ]
                to_ext.append( QgsGeometry.unaryUnion( list_comb ) )
                
            polygon_cleany = [polys[i] for i in range(len(polys)) if i not in to_del]
            polygon_cleany.extend(to_ext)

            
            feedback.pushInfo("writing file...")
            for poly in polygon_cleany:
                f = QgsFeature()
                f.setGeometry(poly)
                dist = 2*np.sqrt(poly.closestSegmentWithContext(poly.centroid().asPoint())[0])
                if dist > 2*rast_pxlX:
                    f.setAttributes([str(dist)])
                    Mask_shp.addFeature(f , QgsFeatureSink.FastInsert)
            
                    
        return{}

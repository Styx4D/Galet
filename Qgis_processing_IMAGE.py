# -*- coding: utf-8 -*-

from re import S
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

import subprocess
try: 
    import rasterio
    from rasterio import Affine as A
except:
    subprocess.check_call(['python', '-m', 'pip', 'install', 'rasterio==1.3.9'])
    import rasterio
    from rasterio import Affine as A

try:
    from PIL import Image
except:
    subprocess.check_call(['python', '-m', 'pip', 'install', 'pillow'])
    from PIL import Image

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

class GALET_image(QgsProcessingAlgorithm):
    
    #INPUT
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_WEIGHT = 'INPUT_WEIGHT'
    
    SCALE_LINE = 'SCALE_LINE'
    SCALE_LEN = 'SCALE_LEN'
    
    #OUTPUT
    OUTPUT_MASK = 'OUTPUT_MASK' 
   
    #config general
    SCALE_RAST = 'SCALE_RAST' 
    CUT_SUPERPOS = 'CUT_SUPERPOS' 
    FILTRE_REC_RESULT = 'FILTRE_REC_RESULT' 
    
    #config mrcnn
    IMAGE_MAX_DIM = 'IMAGE_MAX_DIM'
    DETECTION_MAX_INSTANCES = 'DETECTION_MAX_INSTANCES'
    RPN_NMS_THRESHOLD = 'RPN_NMS_THRESHOLD'
    POST_NMS_ROIS_INFERENCE = 'POST_NMS_ROIS_INFERENCE'
    DETECTION_NMS_THRESHOLD = 'DETECTION_NMS_THRESHOLD'
    PRE_NMS_LIMIT = 'PRE_NMS_LIMIT'
    OUTPUT_RAST = 'OUTPUT_RAST'

    HOST, PORT = "localhost", 9999
    BUFF_SIZE = 65536
    
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
                self.SCALE_RAST,
                self.tr('Scale to use'),
                QgsProcessingParameterNumber.Double,
                defaultValue =2))
                
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
            self.tr('Contour of identified grains')))
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
            self.OUTPUT_RAST,
            self.tr('Scaled Image')))
        
        
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
        n_scale = self.parameterAsDouble(parameters, self.SCALE_RAST, context)
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
        gtl[0] = 0
        _, ytodel = im.size
        gtl[3] = ytodel * scale + 1
        gtl[1] =  scale
        gtl[5] =  -scale
        rast_src.SetGeoTransform(tuple(gtl))
        rast_src = None
        
        (Mask_shp, dest_id) = self.parameterAsSink(parameters, self.OUTPUT_MASK, context, outMask, QgsWkbTypes.MultiPolygon, RAS_IM.crs())
        
        #########################
        #### mrcnn config #######
        #########################
        
        path_h5_file = os.path.abspath(input_weight) #ficher h5 as path
        
        mrcnn_path = str(Path(path_h5_file).parents[2]) #dossier Mask_RCNN

        sys.path.append(mrcnn_path) # mrcnn local
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
        

        # create the grid to infer the image
        feedback.pushInfo("meshing...")
        min_shape = min(bands.shape[:2])

        cut_geom_bb = rast_ext
        cut_width = cut_geom_bb.width()
        cut_height = cut_geom_bb.height()
        
        #top left corner
        x_tlc = cut_geom_bb.xMinimum()
        y_tlc = cut_geom_bb.yMaximum()

        for ns in range(1, int(n_scale) + 1):
             
            len_case_geoX = min_shape / ns * rast_pxlX
            len_case_geoY = min_shape / ns * rast_pxlY
        
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
            x_pxl_tlc = round((x0-x_rast_ext)/rast_pxlX)
            grid_x_pxl = [round(x_pxl_tlc+(gridx[i]-x0)/rast_pxlX) for i in range(len(gridx))]
            
            y_pxl_tlc = round((y_rast_ext-y0)/rast_pxlY)
            grid_y_pxl = [round(y_pxl_tlc+(y0-gridy[i])/rast_pxlY) for i in range(len(gridy))]
            
            #converting to QgsGeometry
            # img, bbox, bbox unoverlap, increment, bbox pxl, extent = x_mini & y_maxi
            sdata= [[[i],[],[],[],[],[]] for i in range(int((ngridx+1)*(ngridy+1)))] 
            nimg=0
            nite=1

            for a in range(0,len(gridy),2+int(ngridx)*ov_iter):
                for i in range(0+a,2+int(ngridx)*ov_iter+a,2):
                    
                    if nite==1:
                        sdata[nimg][1].extend([QgsPointXY(gridx[i],gridy[i]),QgsPointXY(gridx[i+1],gridy[i+1])])
                        sdata[nimg][2].extend([QgsPointXY(gridx[i]+tx_sup_gri*len_case_geoX,gridy[i]-tx_sup_gri*len_case_geoY),\
                        QgsPointXY(gridx[i+1]-tx_sup_gri*len_case_geoX,gridy[i+1]-tx_sup_gri*len_case_geoY)])
                        sdata[nimg][3].extend([i,i+1])
                        sdata[nimg][4].extend([grid_x_pxl[i],grid_x_pxl[i+1],grid_y_pxl[i]])
                        sdata[nimg][5].extend([gridx[i],gridy[i]])
                    else:
                        sdata[nimg][1].extend([QgsPointXY(gridx[i+1],gridy[i+1]),QgsPointXY(gridx[i],gridy[i])])
                        sdata[nimg][2].extend([QgsPointXY(gridx[i+1]-tx_sup_gri*len_case_geoX,gridy[i+1]+tx_sup_gri*len_case_geoY),\
                        QgsPointXY(gridx[i]+tx_sup_gri*len_case_geoX,gridy[i]+tx_sup_gri*len_case_geoY)])
                        sdata[nimg][3].extend([i,i+1])
                        sdata[nimg][4].extend([grid_y_pxl[i+1]])
                    
                    nimg+=1
                
                if nite==1:
                    nimg-=int(ngridx)+1
                    nite=0
                else:
                    nite=1

            if ns ==1:
                data = [sdata[0]]
            else: 
                data.extend(sdata)

        #detecting 
        results=[]
        feedback.pushInfo("MaskR CNN detection...")
        
        def resize_any_depth(arr, new_size):
            accepted_depth = np.min(arr.shape[:2])
            ret = cv.resize(arr[:,:,:accepted_depth], new_size)
            for i in range(accepted_depth,arr.shape[2],accepted_depth):
                ret = np.concatenate((ret, cv.resize(arr[:,:,i:min(i+accepted_depth,arr.shape[2])], new_size)), axis=2)
            return ret
        
        for i in range(len(data)):
            array_d = bands[ data[i][4][2]:data[i][4][3], data[i][4][0]:data[i][4][1], :]
            
            prev_shape = array_d.shape[:2]
            array_d = cv.resize( array_d, (max_dim_infer, max_dim_infer))
            
            feedback.pushInfo(f"image {str(i+1)} on {str(len(data))}. Shape: {str(array_d.shape)}")
            r = self.call_inference(array_d, SA)
            
            mask_array = np.array(r[0]['masks'],dtype=np.uint8)
            r[0]['masks'] = resize_any_depth( mask_array, prev_shape)
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
                    for vec in rasterio.features.shapes(mask_array, mask_array,
                        transform=A.translation(data[id_data][5][0],
                        data[id_data][5][1]) * A.scale(rast_pxlX, -rast_pxlY)):
                        
                        poly.append(vec)
            id_data+=1


        res = ({'geometry': s} for i, (s, _) in enumerate(poly))
        geoms = list(res)

        feedback.pushInfo("converting to QgsGeometry() type")
        geoms_coords =[geoms[i]["geometry"]["coordinates"][0] for i in range(len(geoms))]
        
        polygon_to_clean = [QgsGeometry.fromPolygonXY([[QgsPointXY(geoms_coords[i][j][0],geoms_coords[i][j][1]) \
                            for j in range(len(geoms_coords[i]))]]) for i in range(len(geoms_coords)) ]
        
        #cleaning polygons
        feedback.pushInfo("cleaning "+str(len(polygon_to_clean))+" polygons")
     
        feedback.pushInfo("calculating crossed...")
        id_inter = [[[i,j] for i in range(len(polygon_to_clean)) if polygon_to_clean[i].boundingBoxIntersects(polygon_to_clean[j].boundingBox()) and i!=j]\
                            for j in range(len(polygon_to_clean)) ]
        #flat, sort and set
        id_inter = [item for sublist in id_inter for item in sublist]
        sort_id = [sorted(id_inter[i]) for i in range(len(id_inter)) if any(id_inter[i])]
        sort_id = list(set(tuple(i) for i in sort_id))
         
        feedback.pushInfo("computing IoU")
        to_merge =[sort_id[i][:] for i in range(len(sort_id)) if \
            polygon_to_clean[sort_id[i][0]].intersection(polygon_to_clean[sort_id[i][1]]).area()\
            /polygon_to_clean[sort_id[i][1]].area()>tx_sup_grain or \
            polygon_to_clean[sort_id[i][0]].intersection(polygon_to_clean[sort_id[i][1]]).area()\
            /polygon_to_clean[sort_id[i][0]].area()>tx_sup_grain ]
        
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


        to_del= list(chain.from_iterable(to_merge))
        to_merge_comb = merge_lists_with_common_ids(to_merge)
        to_ext = []
        for list_comb_ids in to_merge_comb:
            list_comb = [ polygon_to_clean[i] for i in list_comb_ids ]
            to_ext.append( QgsGeometry.unaryUnion( list_comb ) )
            
        polygon_cleany = [polygon_to_clean[i] for i in range(len(polygon_to_clean)) if i not in to_del]
        polygon_cleany.extend(to_ext)
        
        feedback.pushInfo("writing file...")
        
        if any(polygon_cleany):
            for poly in polygon_cleany:
                f = QgsFeature()
                f.setGeometry(poly)
                dist = 2*np.sqrt(poly.closestSegmentWithContext(poly.centroid().asPoint())[0])
                f.setAttributes([str(dist)])
                Mask_shp.addFeature(f , QgsFeatureSink.FastInsert)
            
        
        return{}

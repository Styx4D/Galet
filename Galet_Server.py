import numpy as np
import os
import cv2
import sys
import ShareArrays as SA
import pickle
import socketserver

mrcnn_path = "GALET_RCNN_V3/"

#detection module
sys.path.append(mrcnn_path) #librairie mrcnn local
import mrcnn.model as modellib
import grain #config model

from keras.backend import clear_session

path_h5 = "GALET_RCNN_V3/weights/mask_rcnn_granulo_mm_0006.h5"


BUFF_SIZE = 65536
SERV_PORT = 9999



class NetworkMaskRCNNHandler(socketserver.BaseRequestHandler):

	def __init__(self, request, client_address, server, md, sms):
		self.model = md
		self.shared_memory_space = sms
		# super().__init__() must be called at the end
		# because it's immediately calling handle method
		super().__init__(request, client_address, server)

	@classmethod
	def Creator(cls, *args, **kwargs):
		def _HandlerCreator(request, client_address, server):
			cls(request, client_address, server, *args, **kwargs)
		return _HandlerCreator


	def handle(self):
		#read_msg: bytes = b''

		data = []
		while True:
			try:
				packet = self.request.recv(BUFF_SIZE)
				#print(packet)
				if len(packet)<BUFF_SIZE:
					data.append(packet)
					break
				elif not packet:
					break
			except:
				break
			data.append(packet)
			time.sleep(0.001)
			self.request.settimeout(1.)
			#print('.')

		instructions = pickle.loads(b"".join(data))

		if "config" in instructions.keys():
			self.configure(instructions)
		elif "infer" in instructions.keys():
			self.infer(instructions)



	def infer(self, instructions):
		bands, existing_shm = SA.readSharedNPArray(instructions['infer'])

		#detection des mask
		results = self.model.detect([bands], verbose=1)

		particles_number = results[0]['masks'].shape[2]

		print("detected particles number : " + str(particles_number))

		if particles_number==0:
			# we have to set an exception there because the ShareArray system won't work with 0-byte sized objects
			whole_dict = { 'no_detection': True }
			pickled_results = pickle.dumps(whole_dict)
			self.request.sendall(pickled_results)
			return

		print(results[0].keys())

		# sharing the results of the computation to QGis
		shared_msks, shm_msks, dict_msks = SA.shareNPArray(results[0]['masks'])
		shared_rois, shm_rois, dict_rois = SA.shareNPArray(results[0]['rois'])
		shared_cids, shm_cids, dict_cids = SA.shareNPArray(results[0]['class_ids'])
		shared_scrs, shm_scrs, dict_scrs = SA.shareNPArray(results[0]['scores'])

		whole_dict = { 'masks': dict_msks, 'rois': dict_rois, 'class_ids': dict_cids, 'scores': dict_scrs }

		# prevent the server from deleting those once the handle is performed
		self.shared_memory_space.append([shm_msks, shm_rois, shm_cids, shm_scrs])

		pickled_results = pickle.dumps(whole_dict)
		self.request.sendall(pickled_results)



	def configure(self, instructions):
		print("Configuring the model...")

		config_values = instructions['config']

		clear_session()

		config = grain.CustomConfig() #config dans grain.py

		class InferenceConfig(config.__class__):
			IMAGE_MAX_DIM = config_values['IMAGE_MAX_DIM']	 #default 1024
			DETECTION_MAX_INSTANCES = config_values['DETECTION_MAX_INSTANCES']  #Max number of final detections default 100
			RPN_NMS_THRESHOLD = config_values['RPN_NMS_THRESHOLD'] # Non-max suppression threshold to filter RPN proposals. default 0.7
			POST_NMS_ROIS_INFERENCE = config_values['POST_NMS_ROIS_INFERENCE'] # ROIs kept after non-maximum suppression (training and inference) default 1000
			DETECTION_NMS_THRESHOLD = config_values['DETECTION_NMS_THRESHOLD'] # Non-maximum suppression threshold for detection default 0.3
			PRE_NMS_LIMIT = config_values['PRE_NMS_LIMIT'] # ROIs kept after tf.nn.top_k and before non-maximum suppression default 6000
			DETECTION_MIN_CONFIDENCE = 0.001
			IMAGE_RESIZE_MODE = "square"

		config = InferenceConfig()

		model_dir = os.path.join(mrcnn_path, "weights")

		try:
			self.model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
			self.model.load_weights(config_values['path_h5_file'], by_name=True)

			print("\n... Model configuration loaded\n\n")
		except:
			# something wrong happened while loading the model, notify the server
			print("\n----- ERROR -----\n")
			print("Could not load the model weights. Please watch carefully the corresponding paths.\n")
			self.request.sendall(pickle.dumps({'status':'model loading error'}))
			return

		# notify the client that the task has been performed
		self.request.sendall(pickle.dumps({'status':'model loaded'}))



if __name__ == "__main__":
	HOST, PORT = "localhost", SERV_PORT

	clear_session()
	config = grain.CustomConfig() #config dans grain.py
	class_names = ['BG', 'grain_emousse', 'grain_anguleux']
	model_dir = os.path.join(mrcnn_path, "weights")
	
	try:
		model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
	except:
		print("\n----- ERROR -----\n")
		print("Could not load the model, please verify carefully the MRCNN weights path.")
		print("Its expected location is the following: " + model_dir + "\n")
		exit()

	try:
		model.load_weights(path_h5, by_name=True)
	except:
		print("\n----- ERROR -----\n")
		print("Could not load the weights, please verify carefully the path and folder names.")
		print("Weights should be located here: " + path_h5 + "\n")
		exit()


	shared_memory_space = []

	print('Model loaded.')

	# Create the server, binding to localhost on port 9999
	with socketserver.TCPServer((HOST, PORT), NetworkMaskRCNNHandler.Creator(model, shared_memory_space)) as server:
		# Activate the server; this will keep running until you
		# interrupt the program with Ctrl-C

		print('\n\n\n----- GALET Server started. -----\n\nPress Ctrl+C to interrupt the server when you\'re done.\n\n')

		try:
			server.serve_forever()
		except KeyboardInterrupt:
			pass

		print("Goodbye!")
		server.server_close()
		
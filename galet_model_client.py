import numpy as np
import socket
import cv2
import ShareArrays as SA
import pickle



HOST, PORT = "localhost", 9999

BUFF_SIZE = 65536

test_img_path = "D:/GALET_git/galet_pierre/img/cut1.jpg"
#test_img_path = "C:/Users/PierreLemaire/Documents/Styx4D/Cairn/stage Sasha/tests/GOPR4189.jpg"




def exchange_with_server(instructions_dict):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send data
        sock.connect((HOST, PORT))
        #pickled_dict = pickle.dumps()

        print("sending data...")
        sock.sendall(pickle.dumps(instructions_dict))
        print("... data sent")

        data = []
        while True:
            try :
                packet = sock.recv(BUFF_SIZE)
                if len(packet)<BUFF_SIZE:
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


# Load the image and put it into a shared memory space
image = cv2.cvtColor(cv2.imread(test_img_path), cv2.COLOR_BGR2RGB)
imshared, shm, transfer_dict = SA.shareNPArray(image)

# Send it to the server and wait for the response
response_dict = exchange_with_server({'infer':transfer_dict})

# # Create a socket (SOCK_STREAM means a TCP socket)
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
#     # Connect to server and send data
#     sock.connect((HOST, PORT))
#     pickled_dict = pickle.dumps({'infer':transfer_dict})

#     print("sending data...")
#     sock.sendall(pickled_dict)
#     print("... data sent")

#     data = []
#     while True:
#         try :
#             packet = sock.recv(BUFF_SIZE)
#             if len(packet)<BUFF_SIZE:
#                 data.append(packet)
#                 break
#             elif not packet:
#                 break
#         except : 
#             break
#         data.append(packet)
#         time.sleep(0.001)
#         # set it only after the 1st receive so that it wan wait for the computation to happen
#         sock.settimeout(1.)
          
#     received_results = pickle.loads(b"".join(data))

# read every shared array
shared_msks, shm_msks = SA.readSharedNPArray(response_dict['masks'])
shared_rois, shm_rois = SA.readSharedNPArray(response_dict['rois'])
shared_cids, shm_cids = SA.readSharedNPArray(response_dict['class_ids'])
shared_scrs, shm_scrs = SA.readSharedNPArray(response_dict['scores'])

# put them in the correct data structure
result = [{'masks':shared_msks.copy(), 'rois':shared_rois.copy(), 'class_ids':shared_cids.copy(), 'scores':shared_scrs.copy()}]

# release the shared memory
SA.closeSharedNPArrays([shm_msks,shm_rois,shm_cids,shm_scrs], unlink=True)

# print the result to verify that it worked
print(result[0]['masks'].shape)

# release the sent image
SA.closeSharedNPArrays([shm], unlink=True)


#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils


net = jetson.inference.detectNet(network="ssd-mobilenet-v2", threshold=0.5)
#camera = jetson.utils.videoSource("/dev/video0") # '/dev/video0' for V4L2
#display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
#while display.IsStreaming(): # main loop will go here
#	img = camera.Capture()
#	if img is None: # capture timeout
#		print('None')
#		continue
#	detections = net.Detect(img)
#	print(detections)
#	display.Render(img)
#	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

camera = jetson.utils.videoSource("granny_smith_0.jpg")
display = jetson.utils.videoOutput("granny_smith_0_result.jpg") # 'my_video.mp4' for file
img = camera.Capture()

detections = net.Detect(img)
print('detecting..')
display.Render(img)
display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
for detection in detections:
	print(detection)

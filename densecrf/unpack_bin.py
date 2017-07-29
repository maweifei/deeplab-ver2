import struct
import numpy as np
import os
import sys
import cv2
sys.path.insert(0,'mobilenet/utils')
from custom_colormap import applyCustomColorMap

color_map = [[0,0,0],[0,0,255]]
def unpack(bin_file,save_dir):
	'''
	raw data stored in bin_file can be read as bytes string stream. and its format is rows(int) col(int) channels(int) pixels(short)*(rows*cols*channels)
	'''
	with open(bin_file,'rb') as fd:
		rows = struct.unpack('<L',fd.read(4))[0]
		cols = struct.unpack('<L',fd.read(4))[0]
		chas = struct.unpack('<L',fd.read(4))[0]
		total = rows*cols*chas
		pixels = struct.unpack('<{}h'.format(total),fd.read(total*2))
		label = np.array(pixels,dtype=np.uint8)
		label = np.reshape(label,(cols,rows))
		label = label.transpose((1,0))
		print label.shape
		
		label_gray = cv2.cvtColor(label,cv2.COLOR_GRAY2BGR)
		color_label = applyCustomColorMap(label_gray,color_map)
		name,_ = os.path.splitext(bin_file)
		name = name.split('/')[-1]
		print '{}/{}.jpg'.format(save_dir,name)
		cv2.imwrite('{}/{}.jpg'.format(save_dir,name),color_label)
if __name__ == '__main__':
	input_dir= 'save_result'
	save_dir = 'crf_result'
	for input_file in os.listdir(input_dir):
		print 'handing {}.'.format(input_file)
		unpack(os.path.join(input_dir,input_file),save_dir)

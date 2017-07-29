import PIL.Image as Image

if __name__ == '__main__':
	input_dir = 'data'
	output_dir = 'data'
	num = 5
	for i in range(num):
		img = Image.open('{}/{:05d}.jpg'.format(input_dir,i))
		img.save('{}/{:05d}.ppm'.format(output_dir,i))

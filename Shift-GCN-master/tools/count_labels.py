import numpy as np
import pickle as pk


if __name__ == '__main__':
	data_dir = 'data/UAV/skeleton/processed/train_label.pkl'
	cls_num = 155

	with open(data_dir, 'rb') as f:
		samples, label = pk.load(f, encoding='latin1')

	label_count = np.zeros((cls_num, ))
	ls = range(cls_num)

	for i in range(cls_num):
		for j in range(len(label)):
			if label[j] == ls[i]:
				label_count[i] += 1
	print(label_count)



import csv
import numpy as np


def save_result(index, label, save_path, save_name):
	fieldnames = ['index', 'category']
	label = [n for a in label for n in a ]
	name = [n for a in index for n in a ]
	rows = zip(name,label)

	#没有newline=''就会造成隔行写入
	with open(save_path + save_name, 'w',newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(name)):
			writer.writerow({'index': name[i], 'category': label[i]})

def save_label_predict(index, label, predict, save_path, save_name):
	fieldnames = ['index', 'label', 'predict']
	label = [n for a in label for n in a ]
	predict = [n for a in predict for n in a]
	name = [n for a in index for n in a ]
	rows = zip(name, label, predict)

	#没有newline=''就会造成隔行写入
	with open(save_path + save_name, 'w',newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(name)):
			writer.writerow({'index': name[i], 'label': label[i], 'predict': predict[i]})

def save_top5_result(index, prob, name_class, save_path, save_name):


 # index = [n for a in index for n in a ]
 # prob = [n for a in prob for n in a ]

	final = []

	fieldnames = ['top1_cls', 'top2_cls', 'top3_cls','top4_cls', 'top5_cls', 'top1_pro', 'top2_pro', 'top3_pro','top4_pro', 'top5_pro']

	for i in range(name_class):
		# print(i)
		row1 = index[i]
		row2 = prob[i]
		final.append(row1+row2)
	
	# final = [n for a in final for n in a ]

	#没有newline=''就会造成隔行写入
	with open(save_path + save_name, 'w',newline='') as f:
		writer = csv.writer(f)
		for i in range(name_class):
			row = final[i]
			writer.writerow(row)
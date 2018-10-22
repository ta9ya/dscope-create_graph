#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib import rcParams

BAR_IMG = 'bar.png'
RADAR_IMG = 'radar.png'
GRAPH_SIZE = (6, 6)

# rcParams['font.family'] = 'ipagp.tttf'

# fp = FontProperties(fname=r'/Users/iwasa/font/IPAfont00303/ipag.ttf')


def radar_factory(num_vars, frame='circle'):
	"""Create a radar chart with `num_vars` axes.

	This function creates a RadarAxes projection and registers it.

	Parameters
	----------
	num_vars : int
		Number of variables for radar chart.
	frame : {'circle' | 'polygon'}
		Shape of frame surrounding axes.

	"""
	# calculate evenly-spaced axis angles
	theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
	# rotate theta such that the first axis is at the top
	theta += np.pi/2

	def draw_poly_patch(self):
		verts = unit_poly_verts(theta)
		return plt.Polygon(verts, closed=True, edgecolor='k')

	def draw_circle_patch(self):
		# unit circle centered on (0.5, 0.5)
		return plt.Circle((0.5, 0.5), 0.5)

	patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
	if frame not in patch_dict:
		raise ValueError('unknown value for `frame`: %s' % frame)

	class RadarAxes(PolarAxes):

		name = 'radar'
		# use 1 line segment to connect specified points
		RESOLUTION = 1
		# define draw_frame method
		draw_patch = patch_dict[frame]

		def fill(self, *args, **kwargs):
			"""Override fill so that line is closed by default"""
			closed = kwargs.pop('closed', True)
			return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

		def plot(self, *args, **kwargs):
			"""Override plot so that line is closed by default"""
			lines = super(RadarAxes, self).plot(*args, **kwargs)
			for line in lines:
				self._close_line(line)

		def _close_line(self, line):
			x, y = line.get_data()
			# FIXME: markers at x[0], y[0] get doubled-up
			if x[0] != x[-1]:
				x = np.concatenate((x, [x[0]]))
				y = np.concatenate((y, [y[0]]))
				line.set_data(x, y)

		def set_varlabels(self, labels):
			self.set_thetagrids(np.degrees(theta), labels)

		def _gen_axes_patch(self):
			return self.draw_patch()

		def _gen_axes_spines(self):
			if frame == 'circle':
				return PolarAxes._gen_axes_spines(self)
			# The following is a hack to get the spines (i.e. the axes frame)
			# to draw correctly for a polygon frame.

			# spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
			spine_type = 'circle'
			verts = unit_poly_verts(theta)
			# close off polygon by repeating first vertex
			verts.append(verts[0])
			path = Path(verts)

			spine = Spine(self, spine_type, path)
			spine.set_transform(self.transAxes)
			return {'polar': spine}

	register_projection(RadarAxes)
	return theta


def unit_poly_verts(theta):
	"""Return vertices of polygon for subplot axes.

	This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
	"""
	x0, y0, r = [0.5] * 3
	verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
	return verts


def create_bar(rank_data):
	plot_data = rank_data[0:5]
	x = [data['no'] for data in plot_data]
	y = [data['accuracy'] for data in plot_data]
	x_tmp = range(len(x))

	# 棒グラフを作成する。
	fig_bar, ax_bar = plt.subplots(figsize=GRAPH_SIZE)
	# ax_bar = fig.add_subplot(figsize=(4, 4))
	# ax_bar.barh(x_tmp, y)
	# ax_bar.(x_tmp, x)
	ax_bar.barh(x_tmp, y)
	plt.yticks(x_tmp, x)
	# return axes

	# ax = create_bar()
	# axes = create_bar()

	# plt.show()
	plt.savefig(BAR_IMG)


# (axes, ax) = plt.subplots(ncols=2, figsize=(10,4))

# fig, (ax, axes) = plt.subplots(ncols=2, figsize=(10, 4))
# fig = plt.figure()

def create_radarchart(dream_data):
	data = dream_data
	N = 3
	theta = radar_factory(N, frame='polygon')
	Title = 'radar chart'
	#spoke = 'abc'
	spoke_labels = ['kosei', 'yumei', 'zairyoku']
	color = 'b'

	fig, ax = plt.subplots(figsize=GRAPH_SIZE, subplot_kw=dict(projection='radar'))
	# ax = fig.add_subplot(figsize=(5, 5), subplot_kw=dict(projection='radar'))
	fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
	#  chartの範囲
	ax.set_ylim(0, 1)
	# Grid線の位置の指定
	# fig, ax.set_rgrids([0.2, 0.4, 0.6, 0.8])

	fig, ax.set_title(Title, weight='bold', size='medium', position=(0.5, 1.1),
	horizontalalignment='center', verticalalignment='center')

	# 描画処理
	fig, ax.plot(theta, data, color=color)
	fig, ax.fill(theta, data, facecolor=color, alpha=0.50)
	fig, ax.set_varlabels(spoke_labels)

	# 標準のグリッド線は円形なので消す（放射方向だけ残す）
	ax.yaxis.grid(False)
	# 別のグリッドを書く
	ax.plot(theta, [1]*N, 'k-', marker=None, linewidth=0.5, alpha=0.5)
	ax.plot(theta, [2]*N, 'k-', marker=None, linewidth=0.5, alpha=0.5)

	plt.savefig(RADAR_IMG)
		# return ax


def create_graph(result):
	create_bar(result['rank'])
	create_radarchart(result['dream'])
	bar_img = cv2.imread(BAR_IMG)
	radar_img = cv2.imread(RADAR_IMG)

	# print(bar_img.shape)
	# print(radar_img.shape)
	# cv2.imwrite('test_bar.png', bar_img)
	concat_img = cv2.hconcat([bar_img, cv2.resize(radar_img, bar_img.shape[:2])])

	cv2.imwrite('concat.png', concat_img)


if __name__ == '__main__':
	### def create_bar():
	test_data = {'date': '20181016_210221',
				 'list': [0.01181747205555439, 0.00585113326087594, 0.009142509661614895, 0.012987835332751274,
						  0.025216354057192802, 0.014170428737998009, 0.003623287659138441, 0.010311531834304333,
						  0.0166928693652153, 0.015222523361444473, 0.01809479109942913, 0.023610422387719154,
						  0.1897793710231781, 0.017791256308555603, 0.00559137063100934, 0.011362525634467602,
						  0.03674745187163353, 0.017355015501379967, 0.008425896987318993, 0.014723686501383781,
						  0.042360611259937286, 0.030384384095668793, 0.010653607547283173, 0.006072123069316149,
						  0.007502072956413031, 0.015909379348158836, 0.004371880553662777, 0.020194251090288162,
						  0.005607947241514921, 0.00888028834015131, 0.012205892242491245, 0.029918953776359558,
						  0.0024726518895477057, 0.019348012283444405, 0.0032850513234734535, 0.0035030068829655647,
						  0.028869453817605972, 0.017301183193922043, 0.01380877010524273, 0.00812555756419897,
						  0.007828429341316223, 0.0025349927600473166, 0.01325348298996687, 0.0018086005002260208,
						  0.04525774344801903, 0.030202716588974, 0.027985092252492905, 0.006874443031847477,
						  0.009668845683336258, 0.011128797195851803, 0.005976908374577761, 0.013587786816060543,
						  0.003229561261832714, 0.006666489876806736, 0.010049774311482906, 0.005951963365077972,
						  0.010056080296635628, 0.008776976726949215, 0.01986856944859028],
				 'rank': [{'no': '13デザイナー', 'accuracy': 0.1897793710231781},
						  {'no': '45トヨタ社員', 'accuracy': 0.04525774344801903},
						  {'no': '21臨床検査', 'accuracy': 0.042360611259937286},
						  {'no': '17デザイナー', 'accuracy': 0.03674745187163353},
						  {'no': '22臨床検査', 'accuracy': 0.030384384095668793},
						  {'no': '46アイシン社員', 'accuracy': 0.030202716588974},
						  {'no': '32看護師', 'accuracy': 0.029918953776359558},
						  {'no': '37臨床検査技師', 'accuracy': 0.028869453817605972},
						  {'no': '47アイシン社員', 'accuracy': 0.027985092252492905},
						  {'no': '5美容師', 'accuracy': 0.025216354057192802},
						  {'no': '12ケーキ屋', 'accuracy': 0.023610422387719154},
						  {'no': '28臨床検査技師', 'accuracy': 0.020194251090288162},
						  {'no': '59ケーキ屋', 'accuracy': 0.01986856944859028},
						  {'no': '34放射線技師', 'accuracy': 0.019348012283444405},
						  {'no': '11デンソー社員', 'accuracy': 0.01809479109942913},
						  {'no': '14テニスプレイヤ', 'accuracy': 0.017791256308555603},
						  {'no': '18医師', 'accuracy': 0.017355015501379967},
						  {'no': '38看護師', 'accuracy': 0.017301183193922043},
						  {'no': '9演劇(役者)', 'accuracy': 0.0166928693652153},
						  {'no': '26放射線技師', 'accuracy': 0.015909379348158836},
						  {'no': '10演劇(役者)', 'accuracy': 0.015222523361444473},
						  {'no': '20看護師', 'accuracy': 0.014723686501383781},
						  {'no': '6演劇(脚本)', 'accuracy': 0.014170428737998009},
						  {'no': '39看護師', 'accuracy': 0.01380877010524273},
						  {'no': '52プロサッカー選手', 'accuracy': 0.013587786816060543},
						  {'no': '43社長', 'accuracy': 0.01325348298996687},
						  {'no': '4アナウンサー', 'accuracy': 0.012987835332751274},
						  {'no': '31薬剤師', 'accuracy': 0.012205892242491245},
						  {'no': '1デンソー社員', 'accuracy': 0.01181747205555439},
						  {'no': '16ケーキ屋', 'accuracy': 0.011362525634467602},
						  {'no': '50プロ野球選手', 'accuracy': 0.011128797195851803},
						  {'no': '23医師', 'accuracy': 0.010653607547283173},
						  {'no': '8演劇(役者)', 'accuracy': 0.010311531834304333},
						  {'no': '57トヨタ社員', 'accuracy': 0.010056080296635628},
						  {'no': '55ユーチューバー', 'accuracy': 0.010049774311482906},
						  {'no': '49プロ野球選手', 'accuracy': 0.009668845683336258},
						  {'no': '3高校教師', 'accuracy': 0.009142509661614895},
						  {'no': '30臨床検査技師', 'accuracy': 0.00888028834015131},
						  {'no': '58ゲームクリエータ', 'accuracy': 0.008776976726949215},
						  {'no': '19デンソー社員', 'accuracy': 0.008425896987318993},
						  {'no': '40看護師', 'accuracy': 0.00812555756419897},
						  {'no': '41看護師', 'accuracy': 0.007828429341316223},
						  {'no': '25看護師', 'accuracy': 0.007502072956413031},
						  {'no': '48政治家', 'accuracy': 0.006874443031847477},
						  {'no': '54プロサッカー選手', 'accuracy': 0.006666489876806736},
						  {'no': '24看護師', 'accuracy': 0.006072123069316149},
						  {'no': '51プロ野球選手', 'accuracy': 0.005976908374577761},
						  {'no': '56ユーチューバー', 'accuracy': 0.005951963365077972},
						  {'no': '2小学校教師', 'accuracy': 0.00585113326087594},
						  {'no': '29放射線技師', 'accuracy': 0.005607947241514921},
						  {'no': '15テニスプレイヤ', 'accuracy': 0.00559137063100934},
						  {'no': '27薬剤師', 'accuracy': 0.004371880553662777},
						  {'no': '7演劇(役者)', 'accuracy': 0.003623287659138441},
						  {'no': '36放射線技師', 'accuracy': 0.0035030068829655647},
						  {'no': '35薬剤師', 'accuracy': 0.0032850513234734535},
						  {'no': '53プロサッカー選手', 'accuracy': 0.003229561261832714},
						  {'no': '42臨床検査技師', 'accuracy': 0.0025349927600473166},
						  {'no': '33放射線技師', 'accuracy': 0.0024726518895477057},
						  {'no': '44警察官', 'accuracy': 0.0018086005002260208}], 'top': '13デザイナー',
				 'dream': [0.5799465119838715, 0.566234556119889, 0.672523644566536], 'rect': [0, 0, 1.0, 64]}

	create_graph(test_data)

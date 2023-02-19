#####
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use(['ggplot'])
plt.rcParams["legend.fontsize"] = 5
page_width = 6.30045


### robustness
ratio = 0.5
width = page_width * ratio
height = width / 1.618

ax1_data = [("Keyword", [94, 96, 97]),
            ("Syntactic", [92, 96, 95]),
            ("ContextLS", [66, 61, 65])
            ]

ax2_data = [("Keyword", [27, 28, 22]),
            ("Syntactic", [39, 44, 34]),
            ("ContextLS", [0.7, 1.7, 0.5])
            ]
fig, ax = plt.subplots(1,2)

line_alpha = 0.8
for row in ax1_data:
  ax[0].plot(row[1], label=row[0])

for row in ax2_data:
  ax[1].plot(row[1], label=row[0])

xoffset = -0.2
yoffset = 3
anno_fontsize = 7
contextls_value = ax2_data[-1][1]
val = contextls_value[0]
ax[1].annotate(str(val), (0, val+yoffset), fontsize=anno_fontsize)
val = contextls_value[1]
ax[1].annotate(str(val), (0.9, val+yoffset), fontsize=anno_fontsize)
val = contextls_value[2]
ax[1].annotate(str(val), (1.8, val+yoffset), fontsize=anno_fontsize)
#


ax[0].set_xticks([0,1,2], labels=['Del.', "Insert.", "Sub."])
ax[1].set_xticks([0,1,2], labels=['Del.', "Insert.", "Sub."])
ax[0].set_title(r"$\mathcal{R}_{g_{1}}$")
ax[1].set_title(r"$\mathcal{R}_{g_{1}}-\mathcal{R}_{g_{2}}$")
ax[0].legend(fontsize=7)

# plt.rcParams['legend.title_fontsize'] = 'x-small'
# ax[1].legend(loc='lower center', bbox_to_anchor=(1.2, 1.35),
#           fancybox=False, shadow=False, ncol=4, title="Methods")

plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.8, wspace=0.4, hspace=None)
fig.set_size_inches(width, height)
plt.show()
fig.savefig("./visualization/fig/robustness.pdf")


### robustness-bar
ratio = 0.5
width = page_width * ratio
height = width / 1.618

ax1_data = [("Keyword", [94, 96, 97]),
            ("Syntactic", [92, 96, 95]),
            ("ContextLS", [66, 61, 65])
            ]

ax2_data = [("Keyword", [27, 28, 22]),
            ("Syntactic", [39, 44, 34]),
            ("ContextLS", [0.7, 1.7, 0.5])
            ]
fig, ax = plt.subplots(1,2)

line_alpha = 0.8
for row in ax1_data:
  ax[0].bar(range(len(row[1])), row[1], label=row[0])

for row in ax2_data:
  ax[1].bar(range(len(row[1])), row[1], label=row[0])

xoffset = -0.2
yoffset = 3
anno_fontsize = 7
contextls_value = ax2_data[-1][1]
val = contextls_value[0]
ax[1].annotate(str(val), (0, val+yoffset), fontsize=anno_fontsize)
val = contextls_value[1]
ax[1].annotate(str(val), (0.9, val+yoffset), fontsize=anno_fontsize)
val = contextls_value[2]
ax[1].annotate(str(val), (1.8, val+yoffset), fontsize=anno_fontsize)
#


ax[0].set_xticks([0,1,2], labels=['Del.', "Insert.", "Sub."])
ax[1].set_xticks([0,1,2], labels=['Del.', "Insert.", "Sub."])
ax[0].set_title(r"$\mathcal{R}_{g_{1}}$")
ax[1].set_title(r"$\mathcal{R}_{g_{1}}-\mathcal{R}_{g_{2}}$")
ax[0].legend(fontsize=7)

# plt.rcParams['legend.title_fontsize'] = 'x-small'
# ax[1].legend(loc='lower center', bbox_to_anchor=(1.2, 1.35),
#           fancybox=False, shadow=False, ncol=4, title="Methods")

plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.8, wspace=0.4, hspace=None)
fig.set_size_inches(width, height)
plt.show()
fig.savefig("./visualization/fig/robustness-bar.pdf")

## delta robustness
ratio = 0.5
width = page_width * ratio
height = width / 1.618

ax1_data = [("Keyword", [94, 96, 97]),
            ("Syntactic", [92, 96, 95]),
            ("ContextLS", [66, 61, 65])
            ]

ax2_data = [("Keyword", [27, 28, 22]),
            ("Syntactic", [39, 44, 34]),
            ("ContextLS", [0.7, 1.7, 0.5])
            ]
fig, ax = plt.subplots(1,2)

line_alpha = 0.8
for row in ax1_data:
  ax[0].plot(row[1], label=row[0])

for row in ax2_data:
  ax[1].plot(row[1], label=row[0])

xoffset = -0.2
yoffset = 3
anno_fontsize = 7
contextls_value = ax2_data[-1][1]
val = contextls_value[0]
ax[1].annotate(str(val), (0, val+yoffset), fontsize=anno_fontsize)
val = contextls_value[1]
ax[1].annotate(str(val), (0.9, val+yoffset), fontsize=anno_fontsize)
val = contextls_value[2]
ax[1].annotate(str(val), (1.8, val+yoffset), fontsize=anno_fontsize)
#


ax[0].set_xticks([0,1,2], labels=['Del.', "Insert.", "Sub."])
ax[1].set_xticks([0,1,2], labels=['Del.', "Insert.", "Sub."])
ax[0].set_title(r"$\mathcal{R}_{g_{1}}$")
ax[1].set_title(r"$\mathcal{R}_{g_{1}}-\mathcal{R}_{g_{2}}$")
ax[0].legend(fontsize=7)

# plt.rcParams['legend.title_fontsize'] = 'x-small'
# ax[1].legend(loc='lower center', bbox_to_anchor=(1.2, 1.35),
#           fancybox=False, shadow=False, ncol=4, title="Methods")

plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.8, wspace=0.4, hspace=None)
fig.set_size_inches(width, height)
plt.show()
fig.savefig("./visualization/fig/RI_robustness_delta.pdf")

## robustness to types of corruption at higher CR
from numpy import linspace
from matplotlib import cm
ratio = 0.5
width = page_width * ratio
height = width / 1.618

ax1_data = [("2.5", [98, 97, 97]),
            ("5.", [92, 96, 95]),
            ("7.5", [87, 93, 93]),
            ("10.", [83, 92, 92])
            ]

cm = mpl.colormaps['Greens']
start = 0.2
stop = 1
number_of_lines= 4
cm_subsection = linspace(start, stop, number_of_lines)

colors = [cm(x) for x in cm_subsection]
fig, ax = plt.subplots(1,1)

line_alpha = 0.8
for i, row in enumerate(ax1_data):
  ax.plot(row[1], label=row[0], c=colors[i])


ax.set_xticks([0,1,2], labels=['Del.', "Insert.", "Sub."])
ax.set_title(r"$\mathcal{R}_{g_{1}}$ across various CR")
ax.legend(fontsize=7)

# plt.rcParams['legend.title_fontsize'] = 'x-small'
# ax[1].legend(loc='lower center', bbox_to_anchor=(1.2, 1.35),
#           fancybox=False, shadow=False, ncol=4, title="Methods")

plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.8, wspace=0.4, hspace=None)
fig.set_size_inches(width, height)
plt.show()
fig.savefig("./visualization/fig/robustness-at-high-cr.pdf")
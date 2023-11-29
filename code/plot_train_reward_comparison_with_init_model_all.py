# %%
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob
import copy
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from io import BytesIO

parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--plot_model', type = str, required = True)   # plot_model : {'gpt2_small', 'gpt2_small_init_weight=uniform'}
args = parser.parse_args()
my_model = args.plot_model

'''
경로 설정
'''
parent_dir = str(Path(os.getcwd()).parents[0])
RESULT_DIR = parent_dir + '/results'
# REWARD_FILE_DIR_LIST = []
# ACC_FILE_DIR_LIST = []

'''
pdf 파일 불러오기
'''
# pdf_files = glob.glob(RESULT_DIR + '/{}/[!all]*stochastic_quantile_{}.pdf'.format(my_dataset, my_history))
pdf_files = [
            '/home/messy92/Leo/NAS_folder/controllability-of-LM/results/topic-1/{}_comparison_20_epoch_plot_topic-1_stochastic_quantile_acc_history.pdf'.format(my_model),
            '/home/messy92/Leo/NAS_folder/controllability-of-LM/results/topic-1/{}_comparison_20_epoch_plot_topic-1_stochastic_quantile_reward_history.pdf'.format(my_model)
             ]

# List of your PDF files
# pdf_files = [
#     '/home/messy92/Leo/NAS_folder/controllability-of-LM/results/topic-1/gpt2_small_comparison_20_epoch_plot_topic-1_stochastic_quantile_acc_history.pdf',
#     '/home/messy92/Leo/NAS_folder/controllability-of-LM/results/topic-1/gpt2_small_comparison_20_epoch_plot_topic-1_stochastic_quantile_reward_history.pdf',
#     ]
# pdf_files = [
#     '/home/messy92/Leo/NAS_folder/controllability-of-LM/results/topic-1/gpt2_small_init_weight=uniform_comparison_20_epoch_plot_topic-1_stochastic_quantile_acc_history.pdf',
#     '/home/messy92/Leo/NAS_folder/controllability-of-LM/results/topic-1/gpt2_small_init_weight=uniform_comparison_20_epoch_plot_topic-1_stochastic_quantile_reward_history.pdf'
#     ]

'''
전체 범례 설정
'''
# Define labels and colors for the legend (already defined)
# legend_labels = ["q=0.0", "q=0.8", "q=0.9", "q=0.95"]
# legend_colors = ['blue', 'orange', 'red', 'purple']


# # Create custom handles for the legend (already defined)
# custom_handles = [Line2D([0], [0], color=legend_colors[i], marker='o', linestyle='None',
#                          markersize=10, label=legend_labels[i]) for i in range(len(legend_labels))]

'''
Define labels and colors for the legend
'''
legend_labels = ["quantile=0.0", "quantile=0.8", "quantile=0.9", "quantile=0.95"]
legend_colors = ['blue', 'orange', 'green', 'red']

'''
Create custom handles for the legend
'''
custom_handles = [Line2D([0], [0], color=legend_colors[i], marker='o', linestyle='None',
                         markersize=10, label=legend_labels[i]) for i in range(len(legend_labels))]


'''
플롯팅
'''
# Create a figure with 1 row and 4 columns for subplots (already defined)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Adjust the spacing between the subplots (already defined)
plt.subplots_adjust(wspace=0.05, hspace=0)

# Process each PDF file
for i, pdf_file in enumerate(pdf_files):
    # Open the PDF
    doc = fitz.open(pdf_file)

    # Extract the first page (or the page you want)
    page = doc.load_page(0)

    # Get the image of the page at a higher resolution
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))  # Extract at higher resolution

    img = BytesIO(pix.tobytes("png"))

    # Load the high-resolution image into matplotlib
    img_plot = mpimg.imread(img, format='png')

    # Plot the image in the respective subplot
    axs[i].imshow(img_plot)
    axs[i].axis('off')

    # Close the PDF file
    doc.close()

# Add a single legend for the whole figure (already defined)
# fig.legend(handles=custom_handles, loc='lower center', ncol=len(custom_handles), bbox_to_anchor=(0.5, -0.1), fontsize=16)

# Add a single legend for the whole figure to the right of the subplots
plt.legend(handles=custom_handles, bbox_to_anchor=(1.0, 0.9), loc='upper left', fontsize=10)

# Adjust layout (already defined)
plt.tight_layout(rect=[0, 0.01, 1, 1])

# Save the acc_vs_reward plot
if 'init' in my_model:  # if my_model has 'init' substring, i.e., if my_model is an initialized model itself.
    pretraining_setup = 'wo_pretrain'
else:                   # if my_model does not have 'init' substring, i.e., if my_model was initialized with pretrained weights.
    pretraining_setup = 'w_pretrain'    
plt.savefig('/home/messy92/Leo/NAS_folder/controllability-of-LM/results/topic-1/topic-1_acc_vs_reward_plot_{}.pdf'.format(pretraining_setup), dpi=500, bbox_inches='tight')



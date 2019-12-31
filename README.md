# KerasHeaders
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/My\ Drive/Kaggle/Disaster\ NLP\ Model/Kaggle /content/Kaggle
!apt-get install -qq git
!git clone https://github.com/GaryMiles/KerasHeaders.git && mv KerasHeaders/headers.py /content/

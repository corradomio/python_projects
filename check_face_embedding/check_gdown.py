import gdown

url = "https://drive.google.com/file/d/1s-nZMp-LHG0h4dFwvyP_YNBLTijLcrb0/view?usp=share_link"
output = "MSMT17_baseline_RN50_120.pth"


gdown.download(url, output, quiet=False)




import numpy as np
import pywt
import cv2

def dwt_c(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    im=cv2.merge([cA, cH, cV, cD])
    return im

def dwt_c_ones(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    cA=cA/2.0
    cH=(cH+1)/2.0
    cV=(cV+1)/2.0
    cD=(cD+1)/2.0
    im=cv2.merge([cA, cH, cV, cD])
    return im

def dwt_rgb(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    return cA, cH, cV, cD

def dwt_rgb_021(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    cA=cA/2.0
    cH=(cH+1)/2.0
    cV=(cV+1)/2.0
    cD=(cD+1)/2.0
    return cA, cH, cV, cD

def dwt_pre1(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    cA=cA/(2.0*255.0)
    cH=(cH/255.0+1)/2.0
    cV=(cV/255.0+1)/2.0
    cD=(cD/255.0+1)/2.0
    im=cv2.merge([cA, cH, cV, cD])
    return im

def dwt_pre_divide(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    cA=cA/(2.0*255.0)
    cH=(cH/255.0+1)/2.0
    cV=(cV/255.0+1)/2.0
    cD=(cD/255.0+1)/2.0
    return  cA, cH, cV, cD 
    

def idwt(cA, cH, cV, cD):
    cA = cA.cpu().detach().numpy()
    cH = cH.cpu().detach().numpy()
    cV = cV.cpu().detach().numpy()
    cD = cD.cpu().detach().numpy()
    #print(cA.dtype)
    coeffs=cA,(cH,cV,cD)
    im=pywt.idwt2(coeffs, 'haar')
    #print(im.dtype)
    return im 

def idwt_021(cA, cH, cV, cD):
    cA = cA.cpu().detach().numpy()
    cH = cH.cpu().detach().numpy()
    cV = cV.cpu().detach().numpy()
    cD = cD.cpu().detach().numpy()
    cA=cA*2.0
    cH=cH*2.0-1
    cV=cV*2.0-1
    cD=cD*2.0-1
    #print(cA.dtype)
    coeffs=cA,(cH,cV,cD)
    im=pywt.idwt2(coeffs, 'haar')
    #print(im.dtype)
    return im 

def idwt1(cA, cH, cV, cD):
    coeffs=cA,(cH,cV,cD)
    return pywt.idwt2(coeffs, 'haar')

def idwt_pre(cA, cH, cV, cD):
    cA = cA.cpu().detach().numpy()
    cH = cH.cpu().detach().numpy()
    cV = cV.cpu().detach().numpy()
    cD = cD.cpu().detach().numpy()
    cA=cA*(2.0*255.0)
    cH=(cH*2.0-1)*255.0
    cV=(cV*2.0-1)*255.0
    cD=(cD*2.0-1)*255.0
    #print(cA.dtype)
    coeffs=cA,(cH,cV,cD)
    im=pywt.idwt2(coeffs, 'haar')
    #print(im.dtype)
    return im 






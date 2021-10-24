from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):

    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x

def get_covariance(dataset):

    temp = 1/(len(dataset)-1)
    return temp*np.dot(np.transpose(dataset), dataset)

def get_eig(S, m):
    lenS = len(S)
    Lambda, U = eigh(S, eigvals=(lenS-m,lenS-1))
    #reverse
    Lambda = Lambda[::-1]
    #doubt abt formula?
    U = np.transpose(np.transpose(U)[::-1]) 
    return np.diag(Lambda), U

def project_image(img, U):

    return np.dot(np.dot(U,np.transpose(U)),img)

def display_image(orig, proj):
 
    orig = np.reshape(orig,(32,32))
    proj = np.reshape(proj,(32,32))
    orig = np.transpose(orig)
    proj = np.transpose(proj)
    image, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(9,3))
    axis1.set_title("Original")
    axis2.set_title("Projection")
    imgshow1 = axis1.imshow(orig, aspect='equal') 
    imgshow2 = axis2.imshow(proj, aspect='equal')
    image.colorbar(imgshow1, ax=axis1)
    image.colorbar(imgshow2, ax=axis2)
    plt.show()    
    return 

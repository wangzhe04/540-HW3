from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):

        x = np.load(filename)

        meanx = np.mean(x, axis=0)
        x = x - meanx # deduct mean for every x in file, get the center dataset

        return x



def get_covariance(dataset):

    x = np .array(dataset)
    a = 0
    covMatrix = np.dot( np.transpose(x), x) # dot product the transpose of dataset

    for i in x:
        a += 1
    covMatrix = covMatrix/(len(x) - 1) #(1/(n-1) * sum(dotproduct(x, xi))


    return covMatrix


def get_eig(S, m):

    L, U = eigh(S, eigvals=None)


    idx = np.argsort(L)[::-1]
    L = L[idx] #get index
    U = U[:,idx] #get top n elements
    L = L[:m]

    return_vec = np.empty((len(U), m))
    k = 0
    for i in U:
        return_vec[k] = i[:m] #append arr into returnV
        k += 1

    return_val = np.diag(L) #turn to diagnal matrix
    return return_val, return_vec



def get_eig_perc(S, perc):

    L, U = eigh(S, eigvals=None) #get eigenvals and eigenvector
    k = 0

    total_val = 0
    for i in L:
        total_val += i

    for i in L:
        if(i/total_val) > perc:
            k += 1
    return get_eig(S,k)



def project_image(img, U):


    arr = np.array([]) #create an empty np array


    alpha = np.dot(img,U)

    for i in U:
        arr = np.append(arr, np.matmul(alpha, i)) # append the values of alpha*eigenvector for each e_vector to arr


    return arr





def display_image(orig, proj):
    # TODO: add your code here
    a = orig.reshape([32,32], order = 'F')
    b = proj.reshape([32, 32], order = 'F')



    f, (ax1, ax2) = plt.subplots(figsize = (15, 3), ncols = 2)
    a = ax1.imshow(a, aspect = 'equal')
    b = ax2.imshow(b, aspect = 'equal')
    ax1.set_title('Original')
    ax2.set_title('Projection')

    f.colorbar(a, ax=ax1)
    f.colorbar(b, ax=ax2)

    plt.show()


if __name__ == '__main__':

    x = load_and_center_dataset('YaleB_32x32.npy')
    S = get_covariance(x)
    Lambda, U = get_eig(S, 2)
    projection = project_image(x[0], U)
    display_image(x[0], projection)
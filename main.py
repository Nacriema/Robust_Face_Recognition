import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import entropy
from gf import guided_filter
from scipy import ndimage

from matplotlib.image import imsave
from scipy.linalg import lstsq
from scipy.sparse import diags, linalg
from skimage import exposure

from argparse import ArgumentParser


"""
Try following this idea from: 
    * [1] http://warunika.weebly.com/uploads/2/0/5/8/20587050/report_illumination.pdf
    * [2] https://arxiv.org/pdf/1710.05073.pdf
    * [3] https://www2.cs.sfu.ca/~mark/ftp/Eccv04/intrinsicfromentropy.pdf
    * [4] https://www.cs.sfu.ca/~mark/ftp/Iccv03ColorWkshp/iccv03wkshp.pdf
    * [5] https://sci-hub.se/10.1109/TIP.2010.2042645
    * [6] https://sci-hub.se/10.1109/ISDA.2006.253756


The Illumination Invariant Images Algorithm: 

INPUT: RGB image I_origin with shadow

1. Apply Gaussian Smoothing to remove noise

2. Get the chromaticity of I_original as p

3. For theta_k in 1...180
    - Project the data into 1-D space 
    - Find the Entropy nuy_k 

4. Project p in angle theta = argmin_k(nuy) to get the illumination invariant gray scale image I

5. Project back to 2-vector space and ADD BACK LIGHTING INFORMATION to get the illumination invariant chromaticity image
    5.1.
    5.2. Add back lighting information [1] ~ Global Intensity Regularization [2]
        - Notwithstading the illumination normalization, projected-shadow free images may suffer from global intensity diference across images caused by original lighting conditions and by the outliers. 
        - In this step, the most dominant intensity of the resulting image is first approximated by a simple strategy: 
            muy = (mean(Chi(x, y) ^ m)) ^ m  See Contrast Equalization [5]
        - Where m in the regularization coefficient, m = 0.1
        - From [3]: 
            + For display, we would like to move from an intrinsic image, governed by the reflectivity, to one that includes illumination. 
            + So we add back enough e so that the median of the brightness 1% of the pixels has the 2D chromaticity of the original image: 
                Chi_theta -> Chi_theta + Chi_extra_light. Ref [4]
    
6. Locate the shadow mask as napla(I_origin) intercept napla(I)

7. For each color channel in the original image I_origin
    - Set napla(I_origin) = 0 for the pixels belonging to the shadow mask 
    - Solve the Possion equation in Fourier domain to get back the shadow free image I_out
    
OUTPUT: Shadow free 3 band color image I_out
"""

def gaussian_smoothing(img, ksize):
    """
    Use openCV to perform the gaussian smoothing on the image
    Note: This function will work on both RGB and BGR images
    """
    return cv2.GaussianBlur(img, (ksize, ksize), 0) 

def get_chromaticity(brg_img):
    """
    According to: http://warunika.weebly.com/uploads/2/0/5/8/20587050/report_illumination.pdf
    By Finlayson et al:
        p_k = log(c_k) with c_k = R_k / R_p (R_p can bac 3 of RGB)
        
    Note: I think, using get_chromaticity or L1_chromaticity then we have the same effect (May be ?)
    
    BUG: This function may be not good !!! as L1_chromaticity
    """
    R_p = np.power(np.prod(brg_img, axis=-1), (1/3))
    R_p = np.stack((R_p,)*3, axis=-1)
    chromaticity = np.log((brg_img + 1e-8) / (R_p + 1e-8)) # Note: Handle the division errors 
    return chromaticity[:, :, ::-1]

def L1_chromaticity(bgr_img):
    """
    According to: https://moscow.sci-hub.se/803/b01ace080aca675f472708963dd19ec0/finlayson2009.pdf#DREW.INVTCHROM.ICCVWKSHP.03.cite
    The standard definition of chromaticity. ie. colour contents without intensity is defined in an L1 norm
    
    r = {r, g, b} === {R, G, B} / {R + G + B}
    
    Notice: This is the RGB format image
    DONE !!!
    """
    sum_channels = np.stack((np.sum(bgr_img, axis=-1) + 1e-8, )*3, axis=-1)
    chromaticity = bgr_img / sum_channels   # Elementwise division
    return chromaticity[:, :, ::-1] 

def get_rho(brg_img):
    img = cv2.cvtColor(brg_img, cv2.COLOR_BGR2RGB)
    
    r, g, b = cv2.split(img) 
    im_mean = np.mean(img, axis=2)

    mean_r = np.ma.divide(1.*r, im_mean)
    mean_g = np.ma.divide(1.*g, im_mean)
    mean_b = np.ma.divide(1.*b, im_mean)

    log_r = np.ma.log(mean_r)
    log_g = np.ma.log(mean_g)
    log_b = np.ma.log(mean_b)
    
    
    # im_sum = np.sum(img, axis=2)

    # rg_chrom_r = np.ma.divide(1.*r, im_sum)
    # rg_chrom_g = np.ma.divide(1.*g, im_sum)
    # rg_chrom_b = np.ma.divide(1.*b, im_sum)

    # rg_chrom = np.zeros_like(img)

    # rg_chrom[:,:,0] = np.clip(np.uint8(rg_chrom_r*255), 0, 255)
    # rg_chrom[:,:,1] = np.clip(np.uint8(rg_chrom_g*255), 0, 255)
    # rg_chrom[:,:,2] = np.clip(np.uint8(rg_chrom_b*255), 0, 255)
    
    rho = cv2.merge((log_r, log_g, log_b))
    return rho

def minimize_entropy(X, width, height):
    """
    Loop through all posible thetha value, calculate the entropy.
    """
    thetas = np.radians(np.linspace(1, 180, 180))
    
    # BUG: XXX Not sure about this, based on the experiment, I think we need to convert the angle
    cos_sin = np.stack([np.cos(thetas), np.sin(thetas)], axis=0)  # Shape (2, 180)
    entropies = []

    # X shape (h, w, 2), cos_sin shape (2, 180) 
    grayscale_images = X.dot(cos_sin)   # NOTE: This is the equation (14) inside the paper
    print(f'Gray scales shape: {grayscale_images.shape}')
    
    prob = np.array([np.histogram(grayscale_images[..., i], bins='scott', density=True)[0]
                    for i in range(np.size(grayscale_images, axis=-1))], dtype=object)
    entropies = np.array([entropy(p, base=2) for p in prob])
    
    # TODO: ReCheck the last return cos_sin !!! 
    return grayscale_images, entropies, np.argmin(entropies), np.argmax(entropies), cos_sin[:, np.argmin(entropies)], cos_sin[:, np.argmax(entropies)]

def cal_shannon_entropy(img):
    """
    Calculate the entropy of a given 2D gray image
    
    See: https://stackoverflow.com/questions/50313114/what-is-the-entropy-of-an-image-and-how-is-it-calculated
    
    XXX: Bugs - Different results between the numpy version and skimage implementation
    XXX: Notice that the Shannon Entropy criteria in the paper also changed too, maybe I need to see the code_2.py to get the basic idea
    XXX: Overall, I see the result is acceptable !!!, So consider to change this code later !!! 
    """
    # return skimage.measure.shannon_entropy(img)
    marg = np.histogram(np.ravel(img), bins=256)[0] / img.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    entropy = -np.sum(np.multiply(marg, np.log(marg)))
    return entropy

def create_u():
    import math
    
    angle = 120 / 180 * np.pi
    length = np.sqrt(2/3.)
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    first_vect = length * np.array([[0], [1]])
    second_vect =  rot_mat @ first_vect
    third_vect = rot_mat @ second_vect
    
    U = np.hstack((first_vect, second_vect, third_vect))
    
    # Sanity check. OK DONE
    
    U=[[1/math.sqrt(2),-1/math.sqrt(2),0],[1/math.sqrt(6),1/math.sqrt(6),-2/math.sqrt(6)]]
    U=np.array(U)  # Eigens
    return U 

def reduce_chromaticity(chromaticity_img):
    """
    According to: http://warunika.weebly.com/uploads/2/0/5/8/20587050/report_illumination.pdf
    
    Since the chromaticity image p is not 2D, we have to find a projector P which can project
    
    Return:
        - Gray scale Invariant images
        - Entropy of each gray scale images
        - Min index, where the entropy is smallest
    
    """
    u = (np.ones(3)/np.sqrt(3)).reshape(3, -1)
    P_u = np.identity(3) - u @ u.T
    
    # P_u = U.T @ U -> find U ? 
    # See: https://math.stackexchange.com/questions/2194345/given-ata-how-to-find-a
    
    # angle = 120 / 180 * np.pi
    # length = np.sqrt(2/3.)
    # rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
    #                     [np.sin(angle), np.cos(angle)]])
    # first_vect = length * np.array([[0], [1]])
    # second_vect =  rot_mat @ first_vect
    # third_vect = rot_mat @ second_vect
    
    # U = np.hstack((first_vect, second_vect, third_vect))
    
    # # Sanity check. OK DONE
    # print(U)
    # print(U.T @ U)
    # print(U.shape)
    
    U = create_u()
    w, h, _ = chromaticity_img.shape
    # ch_img = chromaticity_img.reshape(-1, 3)  # (N_pixels, 3)
    ch_img = chromaticity_img   # Remain (h, w, 2)
    X = ch_img @ U.T   # (h, w, 3) dot (3, 2) = (h, w, 2)
    
    # X to have shape (h, w, 2)
    print(f'Chi shape: {X.shape}')  
    
    gray_images, entropies, min_idx, max_idx, cos_sin_theta_min, cos_sin_theta_max = minimize_entropy(X, w, h)

    return gray_images, entropies, min_idx, max_idx, cos_sin_theta_min, cos_sin_theta_max, X, w, h

def get_greyscale_back(gray_images, entropies, min_idx):
    """
    NOTE: This step is Optional (IF we use the get_chromaticity function)
    """
    I = gray_images[min_idx]
    return np.exp(I) 

def three_d_vector_representation(gray_images, X, min_idx, cos_sin, width, height):
    """
    From:
        https://www2.cs.sfu.ca/~mark/ftp/Eccv04/intrinsicfromentropy.pdf
    
    3-Vector Representation: 
        
    After we find theta, we can go back to 3D vector representation of points 
    on the projection line via 2x2 projector P_theta:
    1. Form the projected 2D vector X_theta via X_theta = P_theta X
    2. Back to estimate 3D pi via 
    """
    # X has shape (N, 2)
    print(X)
    print(f'Cos sin: {cos_sin}')
    
    # Another idea
    # NOTICE: The angle always > 90 deg

    # e_ = np.array([[-cos_sin[1]], [cos_sin[0]]])  # (-sin, cos)
    
    # NOTE: Use one of 2 line here 
    # e_ = np.array([[cos_sin[1]], [cos_sin[0]]])   # (sin, cos) use when angle < 90
    e_ = np.array([[cos_sin[0]], [cos_sin[1]]])   # (cos, sin) use when angle > 90
    
    # e_ = np.array([[-cos_sin[0]], [cos_sin[1]]])   # (-cos, sin)
    # e_ = cos_sin.reshape(2, 1)
    
    P_et = e_ @ e_.T / (np.linalg.norm(e_))  # Shape (2, 2)
    print(f'My P_et: {P_et}')
    
    # P_et = np.ma.divide(np.dot(e_, e_.T), np.linalg.norm(e_))
    # print(f'P_et value: {P_et}') 
    
    X_theta = X @ P_et  #  (w, h, 2) @ (2, 2) -> (w, h, 2)
    print(f'X_THETA SHAPE: {X_theta.shape}')
    # X_theta = X.T 
    
    # TODO: Add back the X_extra_light to X to match the chromaticities of the brightness pixels in the original image
    print(f'X theta unique value: {np.unique(X_theta)}')
    X_extra_light = find_extra_light(X_theta)
    X_theta /= X_extra_light
    
    U = create_u()
    print(f'U SHAPE: {U.shape}')
    # p = U.T @ X_theta.T  # (3, 2) @ (2, N) = (3, N)
    p = X_theta @ U  # (w, h, 2) @ (2, 3) -> (w, h, 3)

    
    # Convert p from log to normal, code from code_2.py
    mean_estim = np.exp(p)
    estim = np.zeros_like(mean_estim, dtype=np.float64)

    estim[:,:,0] = np.divide(mean_estim[:,:,0], np.sum(mean_estim, axis=2))
    estim[:,:,1] = np.divide(mean_estim[:,:,1], np.sum(mean_estim, axis=2))
    estim[:,:,2] = np.divide(mean_estim[:,:,2], np.sum(mean_estim, axis=2))
    
    plt.imshow(estim)
    plt.savefig('invariant_rg_chromaticity_image.png')
    plt.title('Invariant rg Chromaticity')
    plt.show()

def find_extra_light(X, m=0.1):
    """
    Global Intensity Regularization step
    
    Retreive the most dominant intensity of the resulting image (Chi image) is first approximated by a simple strategy: 
    
        muy = mean(Chi(x, y)^m))^m  with m=0.1
    
    Chi has shape (w, h, 2)
    Muy will have shape (2, )
    """
    muy = np.mean((np.fabs(X) ** m), axis=(0, 1)) ** 1/m
    return muy

def enhance_invariant_rg_chromaticity():
    """
    This is the optional step
    
    In the paper: https://www2.cs.sfu.ca/~mark/ftp/Eccv04/intrinsicfromentropy.pdf
    
    Part 3-Vector Represetation: 
        - For display, we would like to move from an intrinsic image, governed by reflectivity, to one that includes illumination.
    So we add back enough  e so that the median of the brightes 1% of the pixels has the 2D chromaticity of the original image: X_thetha -> X_theta + X_extralight
    """
    # TODO: This is the enhancement, and will be implemented later
    
    pass 

def get_luminance(rgb_img):
    r, g, b = cv2.split(rgb_img)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# Implement the 5.2. Shadow-Specific Edge Detection. DONE
def calibrate_gray_image(unormalize_gray_image):
    """
    Helper function to turn unnormalize gray image to the normal version (inside range 0-1)
    """
    tmp = unormalize_gray_image - np.min(unormalize_gray_image)
    return tmp / np.max(tmp)

def cal_image_gradients(gray_image):
    """
    Calculate the gradient of the grayscale image
    
    Reference: https://stackoverflow.com/questions/49732726/how-to-compute-the-gradients-of-image-using-python
    """
    # Get x-gradient in "sx"
    edges_x = ndimage.sobel(gray_image, axis=0,mode='constant')
    
    # Get y-gradient in "sy"
    edges_y = ndimage.sobel(gray_image,axis=1,mode='constant')
    return edges_x, edges_y

def generate_shadow_specific_edge(gray_invariance_img, gray_light_img, tao_min=0.035, tao_max=0.035):
    """
    Based on Gray Scale Invariance Image (found by mininam theta) and Gray Lighting Image (found by worst theta)
    """
    # TODO: Enhance the gray images by using guided filter to smooth them first
    # Reference: http://www.jiansun.org/papers/GuidedFilter_ECCV10.pdf
    
    r = 8
    eps = 0.001
    
    gray_invariance_img_smoothed = guided_filter(gray_invariance_img, gray_invariance_img, r, eps)
    gray_light_img_smoothed = guided_filter(gray_light_img, gray_light_img, r, eps)
    
    # NOTE: Get the image first derivative along the x and y axis
    
    calibrated_gray_invariance_img = calibrate_gray_image(gray_invariance_img_smoothed)
    calibrated_gray_light_img = calibrate_gray_image(gray_light_img_smoothed)
    
    invariance_grad_x, invariance_grad_y = cal_image_gradients(calibrated_gray_invariance_img)
    light_grad_x, light_grad_y = cal_image_gradients(calibrated_gray_light_img)
    
    # NOTE: 
    phi_min = np.maximum(np.fabs(invariance_grad_x), np.fabs(invariance_grad_y))
    phi_max = np.maximum(np.fabs(light_grad_x), np.fabs(light_grad_y))
    
    print(f'PHI MIN UNIQUE: {np.unique(phi_min)}')
    print(f'PHI MAX UNIQUE: {np.unique(phi_max)}')
    
    plt.imshow(phi_min, cmap='gray')
    plt.savefig('phi_min_image.png')
    plt.show()
    
    plt.imshow(phi_max, cmap='gray')
    plt.savefig('phi_max_image.png')
    plt.show()
    
    mask_1 = phi_min < tao_min
    mask_2 = phi_max > tao_max
    
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(mask_1, cmap='gray')
    axarr[1].imshow(mask_2, cmap='gray')
    plt.savefig('masked_images.png')
    plt.show()
    
    M = np.logical_and(mask_1, mask_2)
    print(M)
    
    plt.title('EDGE MAP')
    plt.imshow(M, cmap='gray')
    plt.savefig('edge_map.png')
    plt.show()
    
    print(f'M SHAPE: {M.shape}')
    return M

# TODO: Implement the 5.3. Full Color Face Image Reconstruction
# This part is the most harder in the overall procedure

# Implement trigger from log-RGB to RGB conversion

def log_transform(rgb_image):
    """
    Function to convert image in form RGB to log-RGB
    """
    print(f'RGB IMAGE: {rgb_image.dtype}')
    log_rgb = np.log1p(rgb_image)  # This is actually I want !!!
    return log_rgb

def inverse_log_transform(log_rgb_image):
    """
    Function to convert back the image from log-RGB to RGB image
    """
    img_origin = np.array(np.expm1(log_rgb_image), dtype=np.uint8)
    return img_origin


def generate_laplace_matrix(size, diagonal_elem=-4, near_elem=1):
    """
    Generate the Laplace matrix based on the paper suggest
    
    This will support for the abstraction level (both scalar and matrix form)
    """
    laplace_matrix = []
    for index in range(size):
        row = [np.zeros_like(diagonal_elem)] * size
        row[index] = diagonal_elem
        if index == 0:
            row[index + 1] = near_elem
        elif index == (size - 1):
            row[index - 1] = near_elem
        else:
            row[index + 1] = near_elem
            row[index - 1] = near_elem
        row = np.block(row)
        laplace_matrix.append(row)
    
    laplace_matrix = np.vstack(laplace_matrix)
    print(f'Laplace matrix: \n{laplace_matrix}')
    return laplace_matrix 

# generate_laplace_matrix(size=3, diagonal_elem=np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), near_elem=np.eye(3))

def generate_v_i(log_image, invert_edge_map):
    grad_x, grad_y = cal_image_gradients(gray_image=log_image)
    sigma_grad_x = np.multiply(grad_x, invert_edge_map)
    sigma_grad_y = np.multiply(grad_y, invert_edge_map)
    v_i = sigma_grad_x + sigma_grad_y
    return v_i

def normalize_log(grayscale_log_image):
    grayscale_log_image -= np.min(grayscale_log_image)
    grayscale_log_image *= 1 / np.max(grayscale_log_image) * np.log(255)
    print(f'UNIQUE VALUE: {np.unique(grayscale_log_image)}')
    return grayscale_log_image

def compute_shadow_free_gradient(L, M):
    """
    Compute shadow-free gradient map ζ for each channel.
    """
    gradients = np.gradient(L)
    grad_x, grad_y = gradients[0], gradients[1]
    
    # Initialize shadow-free gradient map
    ζx = np.where(M == 0, grad_x, 0)
    ζy = np.where(M == 0, grad_y, 0)
    
    return ζx, ζy

def solve_poisson(L, ν):
    """
    Solve Poisson's equation L * Lb = ν.
    """
    Lb = linalg.spsolve(L, ν.flatten())
    return Lb.reshape(ν.shape)

def compute_laplacian(ζx, ζy):
    """
    Compute the Laplacian ν using the shadow-free gradients.
    """
    grad_x, grad_y = np.gradient(ζx), np.gradient(ζy)
    ν = grad_x[0] + grad_y[1]
    
    return ν

def construct_laplacian_matrix(M, N):
    """
    Construct the sparse matrix Λ for Poisson's equation.
    """

    main_diag = -4 * np.ones(M * N)
    side_diag = np.ones(M * N - 1)
    side_diag[np.arange(1, M * N) % N == 0] = 0  # Remove diagonals for boundary conditions

    diagonals = [main_diag, side_diag, side_diag, side_diag, side_diag]
    offsets = [0, -1, 1, -N, N]
    
    L = diags(diagonals, offsets, shape=(M * N, M * N), format='csc')    
    return L

def generate_recovery_image(image_rgb, edge_map):
    """ 
        This is a better version in time processing
    """
    height, width, _ = image_rgb.shape
    # Construct Laplacian matrix
    laplacian_matrix = construct_laplacian_matrix(height, width)

    L = []
    for channel in cv2.split(image_rgb):
        # Apply log transformation
        L_channel = log_transform(channel)
        # Compute shadow-free gradients
        ζx, ζy = compute_shadow_free_gradient(L_channel, edge_map)
        # Compute Laplacian
        ν = compute_laplacian(ζx, ζy)
        # Solve Poisson's equation
        Lb = solve_poisson(laplacian_matrix, ν)
        # Historgram matching from Lb to L
        Lb = exposure.match_histograms(Lb, L_channel)
        # Exponentiate and adjust intensity
        Lb_exp = inverse_log_transform(Lb)
        L.append(Lb_exp)
    reconstructed_img = cv2.merge(L)
    return reconstructed_img

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to image')
    args = parser.parse_args()
    
    img = cv2.imread(args.input)
    plt.imshow(img[:, : ,::-1])
    plt.savefig('original_image.png')
    plt.show() 
    
    print(f"Image shape: {img.shape}")
    # img = gaussian_smoothing(img, ksize=5) 
    # plt.imshow(img[:, :, ::-1])
    # plt.show()
    
    # get_chromaticity(img)
    # rs = L1_chromaticity(img)  
    # plt.imshow(rs)
    # plt.show()
    
    # NOTE: Resize the image for better in process time 
    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    
    
    rs = get_rho(img)
    print(f'UNIQUE VALUE OF RHO: {np.unique(rs)}')
    plt.title('RHO Image')
    plt.imshow(rs)
    plt.savefig('rho_image.png')
    plt.show()
    
    
    
    # TODO: How to convert back from RHO image into RGB image ??? 
    
    # Get the Luminance 
    # lum = get_luminance(rs)
    # plt.imshow(lum, cmap='gray')
    # plt.show()
    
    grays, entropies, min_idx, max_idx, cos_sin_theta_min, cos_sin_theta_max, X, w, h = reduce_chromaticity(rs)
    print(f'Minimum angle: {min_idx + 1}')

    plt.title('Entropy')
    plt.plot(entropies)
    plt.savefig('entropy_minimization.png')
    plt.show()
    
    # X = ch_img @ U.T  (N_pixels, 2) = (N_pixels, 3) @ (3, 2) 
    U = create_u()
    print(f'U VALUE: {U}')

    recon_ch_img = X @ U   #  (w, h, 2) @ (2, 3) = (w, h, 3) 

    plt.title('Reconstruct ch image')
    plt.imshow(recon_ch_img)
    plt.savefig('reconstruct_ch_image.png')
    plt.show()
    
    # for id, _ in enumerate(grays):
    #     plt.title(f'Angle {id + 1}')
    #     plt.imshow(_, cmap='gray')
    #     plt.show()

    
    # rs = get_greyscale_back(rs, rs2, min_idx)
    plt.title('Invariance Grayscale Image (theta min angle)')
    print(f'GRAY MIN: {grays[:, :, min_idx]}')
    plt.imshow(grays[:, :, min_idx], cmap='gray')
    plt.savefig('invariance_grayscale_image.png')
    plt.show()
    
    
    plt.title('Lighting Retained Grayscale Image (theta max angle)')
    plt.imshow(grays[:, :, max_idx], cmap='gray')
    plt.savefig('lighting_retained_grayscale_image.png')
    plt.show()
    
    plt.title('Grayscale Image (theta = 0)')
    plt.imshow(grays[:, :, 0], cmap='gray')
    plt.show()
    
    print(f'Cosin min theta shape: {cos_sin_theta_min.shape}')
    print(f'Cosin max theta shape" {cos_sin_theta_max.shape}')
    # solve_vector_cross_product(grays, X, min_idx)
    
    U = create_u()
    print(f'U is: {U}')
    print(f'U transpose: {U.T}')
    print(f'U @ U.T = {U @ U.T}')
    print(f'U.T @ U = {U.T @ U}')
    
    # Get the chromaticity 
    three_d_vector_representation(grays, X, min_idx, cos_sin_theta_min, w, h)
    
    edge_map = generate_shadow_specific_edge(gray_invariance_img=grays[:, :, min_idx],
                                  gray_light_img=grays[:, :, max_idx], tao_min=0.035, tao_max=0.06)

    recovery_image = generate_recovery_image(image_rgb=img[:, :, ::-1].astype(float), edge_map=edge_map)
    plt.title('Free Log Image')
    
    plt.imshow(recovery_image)
    plt.show()
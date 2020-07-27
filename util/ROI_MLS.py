import numpy as np
import torch
import torch.nn.functional as F

#mls affine inv

def mls_affine_deformation_inv(image, height, width, channel, p, q, alpha=1.0, density=1.0):
    ''' Affine inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    # height = image.shape[0]
    # width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    w[w == np.inf] = 2**31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]

    reshaped_phat = phat.reshape(ctrls, 2, 1, grow, gcol)                               # [ctrls, 2, 1, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    pTwq = np.sum(reshaped_phat * reshaped_w * reshaped_qhat, axis=0)                   # [2, 2, grow, gcol]
    try:
        inv_pTwq = np.linalg.inv(pTwq.transpose(2, 3, 0, 1))                            # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(pTwq.transpose(2, 3, 0, 1))                                 # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(1, 1, grow, gcol)                                    # [1, 1, grow, gcol]
        adjoint = pTwq[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]                        # [2, 2, grow, gcol]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]                  # [2, 2, grow, gcol]
        inv_pTwq = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                       # [grow, gcol, 2, 2]
    mul_left = reshaped_v - qstar                                                       # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)        # [grow, gcol, 1, 2]
    mul_right = np.sum(reshaped_phat * reshaped_w * reshaped_phat2, axis=0)             # [2, 2, grow, gcol]
    reshaped_mul_right =mul_right.transpose(2, 3, 0, 1)                                 # [grow, gcol, 2, 2]
    temp = np.matmul(np.matmul(reshaped_mul_left, inv_pTwq), reshaped_mul_right)        # [grow, gcol, 1, 2]
    reshaped_temp = temp.reshape(grow, gcol, 2).transpose(2, 0, 1)                      # [2, grow, gcol]

    # Get final image transfomer -- 3-D array
    transformers = reshaped_temp + pstar                                                # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf    # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = image[tuple(transformers.astype(np.int16))]    # [grow, gcol]

    # Rescale image
    # transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')

    return transformers.astype(np.float), transformed_image


def mls_affine_deformation_inv_final(height, width, channel, p, q, alpha=1.0, density=1.0):
    ''' Affine inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    # height = image.shape[0]
    # width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    w[w == np.inf] = 2**31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]

    reshaped_phat = phat.reshape(ctrls, 2, 1, grow, gcol)                               # [ctrls, 2, 1, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    pTwq = np.sum(reshaped_phat * reshaped_w * reshaped_qhat, axis=0)                   # [2, 2, grow, gcol]
    try:
        inv_pTwq = np.linalg.inv(pTwq.transpose(2, 3, 0, 1))                            # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(pTwq.transpose(2, 3, 0, 1))                                 # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(1, 1, grow, gcol)                                    # [1, 1, grow, gcol]
        adjoint = pTwq[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]                        # [2, 2, grow, gcol]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]                  # [2, 2, grow, gcol]
        inv_pTwq = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                       # [grow, gcol, 2, 2]
    mul_left = reshaped_v - qstar                                                       # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)        # [grow, gcol, 1, 2]
    mul_right = np.sum(reshaped_phat * reshaped_w * reshaped_phat2, axis=0)             # [2, 2, grow, gcol]
    reshaped_mul_right =mul_right.transpose(2, 3, 0, 1)                                 # [grow, gcol, 2, 2]
    temp = np.matmul(np.matmul(reshaped_mul_left, inv_pTwq), reshaped_mul_right)        # [grow, gcol, 1, 2]
    reshaped_temp = temp.reshape(grow, gcol, 2).transpose(2, 0, 1)                      # [2, grow, gcol]

    # Get final image transfomer -- 3-D array
    transformers = reshaped_temp + pstar                                                # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf    # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    return transformers

def mls_similarity_deformation_inv(image, height, width, channel, p, q, alpha=1.0, density=1.0):
    ''' Similarity inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    w[w == np.inf] = 2**31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]

    mu = np.sum(np.matmul(reshaped_w.transpose(0, 3, 4, 1, 2) * 
                          reshaped_phat1.transpose(0, 3, 4, 1, 2), 
                          reshaped_phat2.transpose(0, 3, 4, 1, 2)), axis=0)             # [grow, gcol, 1, 1]
    reshaped_mu = mu.reshape(1, grow, gcol)                                             # [1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]                                
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
    mul_right = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)       # [ctrls, 2, 2, grow, gcol]
    mul_left = reshaped_qhat * reshaped_w                                               # [ctrls, 1, 2, grow, gcol]
    Delta = np.sum(np.matmul(mul_left.transpose(0, 3, 4, 1, 2), 
                             mul_right.transpose(0, 3, 4, 1, 2)), 
                   axis=0).transpose(0, 1, 3, 2)                                        # [grow, gcol, 2, 1]
    Delta_verti = Delta[...,[1, 0],:]                                                   # [grow, gcol, 2, 1]
    Delta_verti[...,0,:] = -Delta_verti[...,0,:]
    B = np.concatenate((Delta, Delta_verti), axis=3)                                    # [grow, gcol, 2, 2]
    try:
        inv_B = np.linalg.inv(B)                                                        # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(B)                                                          # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(grow, gcol, 1, 1)                                    # [grow, gcol, 1, 1]
        adjoint = B[:,:,[[1, 0], [1, 0]], [[1, 1], [0, 0]]]                             # [grow, gcol, 2, 2]
        adjoint[:,:,[0, 1], [1, 0]] = -adjoint[:,:,[0, 1], [1, 0]]                      # [grow, gcol, 2, 2]
        inv_B = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                          # [2, 2, grow, gcol]

    v_minus_qstar_mul_mu = (reshaped_v - qstar) * reshaped_mu                           # [2, grow, gcol]
    
    # Get final image transfomer -- 3-D array
    reshaped_v_minus_qstar_mul_mu = v_minus_qstar_mul_mu.reshape(1, 2, grow, gcol)      # [1, 2, grow, gcol]
    transformers = np.matmul(reshaped_v_minus_qstar_mul_mu.transpose(2, 3, 0, 1),
                            inv_B).reshape(grow, gcol, 2).transpose(2, 0, 1) + pstar    # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf    # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = image[tuple(transformers.astype(np.int16))]    # [grow, gcol]

    # Rescale image
    # transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')

    return transformers, transformed_image


def mls_rigid_deformation_inv(image, height, width, channel, p, q, alpha=1.0, density=1.0):
    ''' Rigid inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    w[w == np.inf] = 2**31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]

    mu = np.sum(np.matmul(reshaped_w.transpose(0, 3, 4, 1, 2) * 
                          reshaped_phat1.transpose(0, 3, 4, 1, 2), 
                          reshaped_phat2.transpose(0, 3, 4, 1, 2)), axis=0)             # [grow, gcol, 1, 1]
    reshaped_mu = mu.reshape(1, grow, gcol)                                             # [1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]                                
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
    mul_right = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)       # [ctrls, 2, 2, grow, gcol]
    mul_left = reshaped_qhat * reshaped_w                                               # [ctrls, 1, 2, grow, gcol]
    Delta = np.sum(np.matmul(mul_left.transpose(0, 3, 4, 1, 2), 
                             mul_right.transpose(0, 3, 4, 1, 2)), 
                   axis=0).transpose(0, 1, 3, 2)                                        # [grow, gcol, 2, 1]
    Delta_verti = Delta[...,[1, 0],:]                                                   # [grow, gcol, 2, 1]
    Delta_verti[...,0,:] = -Delta_verti[...,0,:]
    B = np.concatenate((Delta, Delta_verti), axis=3)                                    # [grow, gcol, 2, 2]
    try:
        inv_B = np.linalg.inv(B)                                                        # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(B)                                                          # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(grow, gcol, 1, 1)                                    # [grow, gcol, 1, 1]
        adjoint = B[:,:,[[1, 0], [1, 0]], [[1, 1], [0, 0]]]                             # [grow, gcol, 2, 2]
        adjoint[:,:,[0, 1], [1, 0]] = -adjoint[:,:,[0, 1], [1, 0]]                      # [grow, gcol, 2, 2]
        inv_B = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                          # [2, 2, grow, gcol]

    vqstar = reshaped_v - qstar                                                         # [2, grow, gcol]
    reshaped_vqstar = vqstar.reshape(1, 2, grow, gcol)                                  # [1, 2, grow, gcol]

    # Get final image transfomer -- 3-D array
    temp = np.matmul(reshaped_vqstar.transpose(2, 3, 0, 1),
                     inv_B).reshape(grow, gcol, 2).transpose(2, 0, 1)                   # [2, grow, gcol]
    norm_temp = np.linalg.norm(temp, axis=0, keepdims=True)                             # [1, grow, gcol]
    norm_vqstar = np.linalg.norm(vqstar, axis=0, keepdims=True)                         # [1, grow, gcol]
    transformers = temp / norm_temp * norm_vqstar + pstar                               # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf    # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = image[tuple(transformers.astype(np.int16))]    # [grow, gcol]

    # Rescale image
    # transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')

    return transformers, transformed_image

# mls rigid algorithm
def mls_rigid_deformation_inv_wy(image, height, width, channel, p, q, alpha=1.0, density=1.0):
    '''
     Rigid inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = torch.linspace(0, width, steps=int(width * density))
    gridY = torch.linspace(0, height, steps=int(height * density))
    vx, vy = torch.meshgrid(gridY, gridX)

    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    reshaped_v = torch.stack((vx, vy), dim=0)  # [2, grow, gcol]

    w = 1.0 / torch.sum((reshaped_p - reshaped_v) ** 2, dim=1) ** alpha  # [ctrls, grow, gcol]
    w[w == torch.tensor(float("inf"))] = 2 ** 31 - 1
    pstar = torch.sum(w * reshaped_p.permute(1, 0, 2, 3), dim=1) / torch.sum(w, dim=0)  # [2, grow, gcol]
    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    qstar = torch.sum(w * reshaped_q.permute(1, 0, 2, 3), dim=1) / torch.sum(w, dim=0)  # [2, grow, gcol]
    qhat = reshaped_q - qstar  # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]

    neg_phat_verti = phat[:, [1, 0], ...]  # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1, ...] = -neg_phat_verti[:, 1, ...]
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    mul_right = torch.cat((reshaped_phat1, reshaped_neg_phat_verti), dim=1)  # [ctrls, 2, 2, grow, gcol]
    mul_left = reshaped_qhat * reshaped_w  # [ctrls, 1, 2, grow, gcol]
    Delta = torch.sum(torch.matmul(mul_left.permute(0, 3, 4, 1, 2),
                             mul_right.permute(0, 3, 4, 1, 2)),
                   dim=0).permute(0, 1, 3, 2)  # [grow, gcol, 2, 1]
    Delta_verti = Delta[..., [1, 0], :]  # [grow, gcol, 2, 1]
    Delta_verti[..., 0, :] = -Delta_verti[..., 0, :]
    B = torch.cat((Delta, Delta_verti), dim=3)  # [grow, gcol, 2, 2]
    try:
        inv_B = torch.inverse(B)  # [grow, gcol, 2, 2]
        flag = False
    except:
        flag = True
        det = np.linalg.det(B.numpy())
        det = torch.from_numpy(det)  # [grow, gcol]
        det[det < 1e-8] = torch.tensor(float("inf"))
        reshaped_det = det.reshape(grow, gcol, 1, 1)  # [grow, gcol, 1, 1]
        adjoint = B[:, :, [[1, 0], [1, 0]], [[1, 1], [0, 0]]]  # [grow, gcol, 2, 2]
        adjoint[:, :, [0, 1], [1, 0]] = -adjoint[:, :, [0, 1], [1, 0]]  # [grow, gcol, 2, 2]
        inv_B = (adjoint / reshaped_det).permute(2, 3, 0, 1)  # [2, 2, grow, gcol]

    vqstar = reshaped_v - qstar  # [2, grow, gcol]
    reshaped_vqstar = vqstar.reshape(1, 2, grow, gcol)  # [1, 2, grow, gcol]

    # Get final image transfomer -- 3-D array
    temp = torch.matmul(reshaped_vqstar.permute(2, 3, 0, 1),
                     inv_B).reshape(grow, gcol, 2).permute(2, 0, 1)  # [2, grow, gcol]
    norm_temp = torch.norm(temp, dim=0, keepdim=True)  # [1, grow, gcol]
    norm_vqstar = torch.norm(vqstar, dim=0, keepdim=True)  # [1, grow, gcol]
    transformers = temp / (norm_temp + 1e-10) * norm_vqstar + pstar  # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == torch.tensor(float("inf"))  # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image

    transformed_image = image[tuple(transformers.numpy().astype(np.int16))]  # [grow, gcol]

    # # Rescale image
    # img_h, img_w, _ = transformed_image.shape
    # transformed_image = transformed_image.resize_(int(img_h/density), int(img_w/density), channel)

    return transformers.numpy(), transformed_image








# mls for whole feature, instead of roi align
def roi_mls_whole(feature, d_point, g_point, step=1):
    '''
    :param feature: itorchut guidance feature [C, H, W]
    :param d_point: landmark for degraded feature [N, 2]
    :param g_point: landmark for guidance feature [N, 2]
    :param step: step of landmark choose, number of control points: landmark_number/step
    :return: transformed feature [C, H, W]
    '''
    # feature 3 * 256 * 256
    
    channel = feature.size(0)
    height = feature.size(1) # 256 * 256
    width = feature.size(2)

    # ignore the boarder point of face
    g_land = g_point[0::step, :]
    d_land = d_point[0::step, :]

    # mls
    
    featureTmp = feature.permute(1,2,0)
    # grid, timg = mls_rigid_deformation_inv_wy(featureTmp.cpu(), height, width, channel, g_land.cpu(), d_land.cpu(), density=1.) # 2 * 256 * 256 # wenyu 
    grid, timg = mls_affine_deformation_inv(featureTmp.cpu(), height, width, channel, g_land.cpu().numpy(), d_land.cpu().numpy(), density=1.) #affine prefered
    # grid, timg_sim = mls_similarity_deformation_inv(featureTmp.cpu(), height, width, channel, g_land.cpu().numpy(), d_land.cpu().numpy(), density=1.) # similarity
    # grid, timg_rigid = mls_rigid_deformation_inv(featureTmp.cpu(), height, width, channel, g_land.cpu().numpy(), d_land.cpu().numpy(), density=1.) #rigid


    grid = (grid - 127.5) / 127.5
    gridNew = torch.from_numpy(grid[[1,0],:,:]).float().permute(1,2,0).unsqueeze(0)

    if torch.cuda.is_available():
        gridNew = gridNew.cuda()
    featureNew = feature.unsqueeze(0)
    # warp_feature = F.grid_sample(featureNew,gridNew.cuda(),mode='nearest')
    warp_feature = F.grid_sample(featureNew,gridNew.cuda())

    # HWC -> CHW
    

    # _, timg_affine = mls_affine_deformation_inv(featureTmp.cpu(), height, width, channel, g_land.cpu().numpy(), d_land.cpu().numpy(), density=1.)
    # _, timg_sim = mls_similarity_deformation_inv(featureTmp.cpu(), height, width, channel, g_land.cpu().numpy(), d_land.cpu().numpy(), density=1.)
    # _, timg_rigid = mls_rigid_deformation_inv(featureTmp.cpu(), height, width, channel, g_land.cpu().numpy(), d_land.cpu().numpy(), density=1.)

    # return warp_feature.squeeze(), timg.permute(2,0,1) #, timg_sim.permute(2,0,1), timg_rigid.permute(2,0,1)
    return warp_feature.squeeze(), gridNew.cuda()


def roi_mls_whole_final(feature, d_point, g_point, step=1):
    '''
    :param feature: itorchut guidance feature [C, H, W]
    :param d_point: landmark for degraded feature [N, 2]
    :param g_point: landmark for guidance feature [N, 2]
    :param step: step of landmark choose, number of control points: landmark_number/step
    :return: transformed feature [C, H, W]
    '''
    # feature 3 * 256 * 256
    
    channel = feature.size(0)
    height = feature.size(1) # 256 * 256
    width = feature.size(2)

    # ignore the boarder point of face
    g_land = g_point[0::step, :]
    d_land = d_point[0::step, :]

    # mls
    
    # featureTmp = feature.permute(1,2,0)
    # grid, timg = mls_rigid_deformation_inv_wy(featureTmp.cpu(), height, width, channel, g_land.cpu(), d_land.cpu(), density=1.) # 2 * 256 * 256 # wenyu 
    grid = mls_affine_deformation_inv_final(height, width, channel, g_land.cpu().numpy(), d_land.cpu().numpy(), density=1.) #affine prefered
    # grid, timg_sim = mls_similarity_deformation_inv(featureTmp.cpu(), height, width, channel, g_land.cpu().numpy(), d_land.cpu().numpy(), density=1.) # similarity
    # grid, timg_rigid = mls_rigid_deformation_inv(featureTmp.cpu(), height, width, channel, g_land.cpu().numpy(), d_land.cpu().numpy(), density=1.) #rigid

    grid = (grid - height/2) / (height/2)

    gridNew = torch.from_numpy(grid[[1,0],:,:]).float().permute(1,2,0).unsqueeze(0)

    if torch.cuda.is_available():
        gridNew = gridNew.cuda()
    # HWC -> CHW
    return gridNew
#  Copyright (c) 2019. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import numpy as np
from math import pi, exp
import scipy.linalg
import cv2


class Reconstruction:
    """
    Zmiany modułu rekonstrukcji na dzień:
    Coś o parametrach
    Jakie algorytmy rekonstrukcji są dostępne...

    """

    lambda_parameter = 19000

    def __init__(self,
                 effect_matrix=None,
                 background_vector=None,
                 model_visibility=None,
                 visibility_outerior=None,
                 fspl = None):
        '''
        :param effect_matrix:
        :param background_vector:
        :param model_visibility:
        :param visibility_outerior:
        :param fspl:
        '''

        self.effect_matrix = np.genfromtxt(effect_matrix, delimiter=',')
        self.background_vector = self.__m2v(np.genfromtxt(background_vector, delimiter=','))
        self.model_visibility = np.genfromtxt(model_visibility, delimiter=',')
        self.visibility_outerior = np.genfromtxt(visibility_outerior, delimiter=',')
        self.fspl = self.__m2v(np.genfromtxt(fspl, delimiter=','))

    def tikhonov_regularization(self, measure_vector):
        '''
        :param measure_vector:
        :return:
        '''
        measure_vector = self.set_measure_vector(measure_vector)
        F_tikhonov = abs(measure_vector - self.background_vector)
        tikhonov_helper = (np.dot(self.effect_matrix, self.effect_matrix.transpose())).shape
        tikhonov_temp = (np.mat(self.effect_matrix) * np.mat(self.effect_matrix.transpose())
                         + self.lambda_parameter * np.eye(tikhonov_helper[0], tikhonov_helper[1]))
        tikhonov_temp2 = np.linalg.solve(tikhonov_temp, F_tikhonov)
        tikhonov_temp3 = np.reshape(tikhonov_temp2, (120, 1))

        tikhonov = np.mat(self.effect_matrix.transpose()) * np.mat(tikhonov_temp3)

        return tikhonov

    def svd_regularization(self, measure_vector):
        '''
        :param measure_vector:
        :return:
        '''
        measure_vector = self.set_measure_vector(measure_vector)
        F_svd = abs(measure_vector - self.background_vector) + 1.501
        F_svd = np.reshape(F_svd, (120, 1), order='F')
        svd_temp = np.linalg.pinv(self.effect_matrix, rcond=0.250141)
        svd = np.mat(svd_temp) * np.mat(F_svd)
        return svd

    def svd_regularization_two(self, measure_vector):
        '''
        :param measure_vector:
        :return:
        '''
        measure_vector = self.set_measure_vector(measure_vector)
        F_svd = abs(measure_vector - self.background_vector) + 1.501
        F_svd = np.reshape(F_svd, (120, 1))
        svd_temp = scipy.linalg.pinv(self.effect_matrix, rcond=0.15)
        svd = np.mat(svd_temp) * np.mat(F_svd)
        return svd

    def preparing(self, reconstruction):
        '''
        :param reconstruction:
        :return:
        '''
        MOD_fov = np.zeros([65, 63])
        RECON = MOD_fov
        RECON = np.matrix.flatten(RECON, order='F')

        # reading model visibility
        #df = pd.read_csv('const/model/mod_fovIdx_room.csv', header=None)
        # MOD_fovIdx = df.values
        size_model_vis = len(self.model_visibility)
        MOD_fovIdx = np.reshape(self.model_visibility,(size_model_vis,1),order='F').astype(int)
        MOD_fovIdx = MOD_fovIdx - 1

        # preparing space for matrix
        RECON[MOD_fovIdx] = reconstruction
        x = np.reshape(RECON, (65, 63), order='F')
        gg = self.__mex_hat_filtr(3, 3)

        # reading visibility outerior
        # df = pd.read_csv('const/model/mod_outerior_room.csv', header=None)
        # MOD_outerior = df.values
        size_vis_out = len(self.visibility_outerior)
        MOD_outerior = np.reshape(self.visibility_outerior,(size_vis_out,1),order='F').astype(int)
        MOD_outerior = MOD_outerior - 1

        # rotation, filtration, normalization
        x = np.flip(np.rot90(x))
        dst = cv2.filter2D(x, -1, gg)

        out = np.zeros(dst.shape, np.double)
        normalized = cv2.normalize(dst, out, 1.0, 0.0, cv2.NORM_MINMAX)
        normalized[normalized < 0.75] = 0

        # making a circle
        normalized = np.matrix.flatten(normalized, order='C')

        # deleteing outerior
        normalized[MOD_outerior] = 1
        normalized = np.reshape(normalized, (65, 63), order='F')

        return normalized#np.rot90(np.flipud(normalized), -1)

    def __mex_hat_filtr(self, n, sig):
        pom_n = 2 * n + 1
        mex_hat = np.zeros([pom_n, pom_n])
        for xd in range(-n, n + 1):
            for yd in range(-n, n + 1):
                fic = (np.power(xd, 2) + np.power(yd, 2)) / (2 * sig ** 2)
                xi = xd + n
                yi = yd + n
                mex_hat[xi, yi] = (1 / pi * sig ** 2) * (1 - fic) * exp(-fic)

        return mex_hat

    def __m2v(self, matrix):
        np.fill_diagonal(matrix, 0)
        u_temp = np.triu(matrix)
        l_temp = np.tril(matrix)
        temp_meas = (np.transpose(u_temp) + l_temp) * 0.5
        vec_meas = np.matrix.flatten(temp_meas, order='F')  # matlab style z Fortrana
        vec_meas = vec_meas[np.nonzero(vec_meas)]
        meas_vector = abs(vec_meas)

        return np.array(meas_vector)

    def set_lambda_parameter(self, lambda_value):
        self.lambda_parameter = lambda_value

    def set_measure_vector(self, m_vector):
        return self.__m2v(np.genfromtxt(m_vector, delimiter=','))

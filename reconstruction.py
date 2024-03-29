import cv2
import numpy as np
import scipy.linalg
import scipy.signal
from math import pi, exp


class Reconstruction:
    """
    Jakiś opis modułu
    Coś o parametrach
    Jakie algorytmy rekonstrukcji są dostępne...

    """

    lambda_parameter = 19000

    def __init__(self,
                 effect_matrix=None,
                 background_vector=None,
                 model_visibility=None,
                 visibility_outerior=None,
                 fspl=None):
        """
        :param effect_matrix:
        :param background_vector:
        :param model_visibility:
        :param visibility_outerior:
        :param fspl:
        """

        self.effect_matrix = np.genfromtxt(effect_matrix, delimiter=',')
        #self.background_vector = self.__m2v(np.genfromtxt(background_vector, delimiter=','))
        self.background_vector = (np.genfromtxt(background_vector, delimiter=','))
        self.model_visibility = np.genfromtxt(model_visibility, delimiter=',')
        self.visibility_outerior = np.genfromtxt(visibility_outerior, delimiter=',')
        self.fspl = self.__m2v(np.genfromtxt(fspl, delimiter=','))

    def tikhonov_regularization(self, measure_vector):
        """
        :param measure_vector:
        :return:
        """
        #measure_vector = self.set_measure_vector(measure_vector)
        temp = abs(self.background_vector-measure_vector)+1.501
        f_tikhonov = self.set_measure_vector(temp)#abs(measure_vector - self.background_vector)

        tikhonov_helper = (np.dot(self.effect_matrix, self.effect_matrix.transpose())).shape
        tikhonov_temp = (np.mat(self.effect_matrix) * np.mat(self.effect_matrix.transpose())
                         + self.lambda_parameter * np.eye(tikhonov_helper[0], tikhonov_helper[1]))
        tikhonov_temp2 = np.linalg.solve(tikhonov_temp, f_tikhonov)
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
        """
        :param reconstruction:
        :return:
        """
        mod_fov = np.zeros([65, 63])
        recon = mod_fov
        recon = np.matrix.flatten(recon, order='F')

        # reading model visibility
        # df = pd.read_csv('const/model/mod_fovIdx_room.csv', header=None)
        # mod_fov_idx = df.values
        size_model_vis = len(self.model_visibility)
        mod_fov_idx = np.reshape(self.model_visibility, (size_model_vis, 1), order='F').astype(int)
        mod_fov_idx = mod_fov_idx - 1

        # preparing space for matrix
        recon[mod_fov_idx] = reconstruction
        x = np.reshape(recon, (65, 63), order='F')
        gg = self.__mex_hat_filtr(3, 7)

        # reading visibility outerior
        # df = pd.read_csv('const/model/mod_outerior_room.csv', header=None)
        # mod_outerior = df.values
        size_vis_out = len(self.visibility_outerior)
        mod_outerior = np.reshape(self.visibility_outerior, (size_vis_out, 1), order='F').astype(int)
        mod_outerior = mod_outerior - 1

        #rotation, filtration, normalization
        # wywalenie rotacji i filtra2D
        #x = np.flip(np.rot90(x))
        dst = cv2.filter2D(x, -1, gg)
        #dst = scipy.signal.convolve2d(x,gg)

        out = np.zeros(dst.shape, np.double)
        normalized = cv2.normalize(dst, out, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F) # dodany dtype
        normalized[normalized < 0.9] = 0

        # making a circle
        #normalized: np.ndarray = np.matrix.flatten(normalized, order='F')
        normalized = np.matrix.flatten(normalized, order='F')
        # deleteing outerior
        normalized[mod_outerior] = 1
        normalized = np.reshape(normalized, (65, 63), order='F')

        return normalized#np.rot90(np.flipud(normalized), -1)

    @staticmethod
    def __mex_hat_filtr(n, sig):
        pom_n = 2 * n + 1
        mex_hat = np.zeros([pom_n, pom_n])
        for xd in range(-n, n + 1):
            for yd in range(-n, n + 1):
                fic = (np.power(xd, 2) + np.power(yd, 2)) / (2 * sig ** 2)
                xi = xd + n
                yi = yd + n
                mex_hat[xi, yi] = (1 / pi * sig ** 2) * (1 - fic) * exp(-fic)

        return mex_hat

    @staticmethod
    def __m2v(matrix):
        for x in range(0, 15):
            for y in range(0, 15):
                if matrix[x, y] >= 240 or matrix[x, y] == 0:
                    matrix[x, y] = matrix[x, y-1]
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
        # In case of reading file
        # return self.__m2v(np.genfromtxt(m_vector, delimiter=','))
        return self.__m2v(np.asarray(m_vector))

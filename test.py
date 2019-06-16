import reconstruction as rec
import matplotlib.pyplot as plt

'''
:param effect_matrix_file: Parametry...
:param background_vector_file: Parametry tła ...
:param model_visibility_file: 
:param visibility_outerior_file: 
:param fspl_file:
'''

effect_matrix_file = 'const/model/effect_matrix_room.csv'
background_vector_file = 'const/measurements/Frame_2.csv'
model_visibility_file = 'const/model/mod_fovIdx_room.csv'
visibility_outerior_file = 'const/model/mod_outerior_room.csv'
fspl_file = 'const/physical_constant/fspl.csv'


measure_vector_file = 'const/measurements/Frame_15.csv'

rekonstrukcja = rec.Reconstruction(effect_matrix=effect_matrix_file,
                                   background_vector=background_vector_file,
                                   model_visibility=model_visibility_file,
                                   visibility_outerior=visibility_outerior_file,
                                   fspl=fspl_file)

tickonov = rekonstrukcja.tikhonov_regularization(measure_vector_file)
tikonov_norm = rekonstrukcja.preparing(tickonov)

plt.figure(1)
plt.imshow(tikonov_norm)
# hidding x,y axis
frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)
# hiding pics boundary
frame1.spines['top'].set_visible(False)
frame1.spines['right'].set_visible(False)
frame1.spines['bottom'].set_visible(False)
frame1.spines['left'].set_visible(False)
# show colorbar
plt.colorbar()

plt.show()

#Parte 1: Importar informacion

#Importar modulos de Python
import mne
from pathlib import Path

#Descargar datos de un EEG de ejemplo
data = Path("data/test_raw.fif")
raw = mne.io.read_raw_fif(data)

#Los datos se leen crudos
print(raw)
print(raw.info)

#Se grafican los datos crudos debajo de los 50hz de frecuencia
raw.plot_psd(fmax=50)
raw.plot(duration=5, n_channels=30)

#Parte 2: Preprocesamiento

#Se preprocesa la informacion a partir de análisis de componentes independientes (ICA)
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]
ica.plot_properties(raw, picks=ica.exclude)

#Se crea una copia de los datos crudos
orig_raw = raw.copy()
raw.load_data()
ica.apply(raw)

#Se seleccionan canales especificos para su contraste
chs = ['MEG 0111', 'MEG 0121', 'MEG 0131', 'MEG 0211', 'MEG 0221', 'MEG 0231',
       'MEG 0311', 'MEG 0321', 'MEG 0331', 'MEG 1511', 'MEG 1521', 'MEG 1531',
       'EEG 001', 'EEG 002', 'EEG 003', 'EEG 004', 'EEG 005', 'EEG 006',
       'EEG 007', 'EEG 008']
chan_idxs = [raw.ch_names.index(ch) for ch in chs]
orig_raw.plot(order=chan_idxs, start=12, duration=4)
raw.plot(order=chan_idxs, start=12, duration=4)



#Parte 3: Diferenciar eventos: estimulos

#Mostrar primeros 5 eventos detectados del canal designado
events = mne.find_events(raw, stim_channel='STI 014')
print(events[:5])

#Nombrar los eventos y asignarles un valor
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'smiley': 5, 'buttonpress': 32}
fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp)



#Parte 4: Filtracion de datos a traves de epocas

# Rechazar datos con una amplitud mayor a lo razonable
reject_criteria = dict(mag=4000e-15,     # 4000 fT
                       grad=4000e-13,    # 4000 fT/cm
                       eeg=150e-6,       # 150 µV
                       eog=250e-6)       # 250 µV
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.5,
                    reject=reject_criteria, preload=True)

#Segmentar señales, entre visuales y auditivas
conds_we_care_about = ['auditory/left', 'auditory/right',
                       'visual/left', 'visual/right']
epochs.equalize_event_counts(conds_we_care_about)
aud_epochs = epochs['auditory']
vis_epochs = epochs['visual']
del raw, epochs

#Preparar canales especificos
aud_epochs.plot_image(picks=['MEG 1332', 'EEG 021'])



#Parte 5: Estimacion de epocas

#Sacar un promedio de ambas epocas
aud_evoked = aud_epochs.average()
vis_evoked = vis_epochs.average()
mne.viz.plot_compare_evokeds(dict(auditory=aud_evoked, visual=vis_evoked),
                             legend='upper left', show_sensors='upper right')

#Cargar los promedios
aud_evoked.plot_joint(picks='eeg')
aud_evoked.plot_topomap(times=[0., 0.08, 0.1, 0.12, 0.2], ch_type='eeg')

#Contrastar datos de los promedios
evoked_diff = mne.combine_evoked([aud_evoked, vis_evoked], weights=[1, -1])
evoked_diff.pick_types(meg='mag').plot_topo(color='r', legend=False)



#Parte 6: Estimar los orígenes de la actividad

#Cargamos el operador inverso precargado
inverse_operator_file = (sample_data_folder / 'MEG' / 'sample' /
                         'sample_audvis-meg-oct-6-meg-inv.fif')
inv_operator = mne.minimum_norm.read_inverse_operator(inverse_operator_file)

#Establecer datos de relacion entre señal y ruido
snr = 3.
lambda2 = 1. / snr ** 2

#Caragamos el STC
stc = mne.minimum_norm.apply_inverse(vis_evoked, inv_operator,
                                     lambda2=lambda2,
                                     method='MNE')

#Trazamos direccion de los sujetos cargados
subjects_dir = sample_data_folder / 'subjects'

#FGraficamos el STC
stc.plot(initial_time=0.1, hemi='split', views=['lat', 'med'],
         subjects_dir=subjects_dir)

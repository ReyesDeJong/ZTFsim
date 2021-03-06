import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import h5py
import re

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)

import parameters.general_keys as general_keys
import parameters.dataframe_keys as dataframe_keys


def save_pickle(data, path):
  pkl.dump(data, open(path, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)


class SimulatedData2Dataframe(object):

  def __init__(self, path_simulated_data, path_real_dataframe, verbose=False):
    self.path_real_dataframe = path_real_dataframe
    self.path_simulated_data = path_simulated_data
    self.bands_to_num_dict = {'g': 1, 'r': 2}
    # self.labels_to_num_dict = {}
    self.verbose = verbose

  def atoi(self, text):
    return int(text) if text.isdigit() else text

  def natural_keys(self, text):
    return [self.atoi(c) for c in re.split(r'(\d+)', text)]

  def human_sort(self, numpy_array):
    array_as_list = list(numpy_array)
    array_as_list.sort(key=self.natural_keys)
    return np.array(array_as_list)

  def load_dataframe(self):
    return pd.read_pickle(self.path_real_dataframe)

  def load_simulated_data(self):
    return h5py.File(self.path_simulated_data, "r")

  def _get_empty_dict_for_simulated_data(self):
    dict_to_contain_simulated_data = {
      dataframe_keys.OID: [], dataframe_keys.RA: [], dataframe_keys.DEC: [],
      dataframe_keys.FID: [], dataframe_keys.JD: [],
      dataframe_keys.MAGPSF_CORR: [], dataframe_keys.SIGMAPSF_CORR: [],
      dataframe_keys.LC_TYPE: [], dataframe_keys.DETECTED: []}
    return dict_to_contain_simulated_data

  def _get_n_objects_ra_dec_data_frames(self, n_oids=1000):
    return None
    # oids_g = np.unique(df[df[fid_key] == bands_to_num_dict[g_key]][oid_key])[
    #          :1000]
    # oids_r = np.unique(df[df[fid_key] == bands_to_num_dict[r_key]][oid_key])[
    #          :1000]
    # oid_df_g = df[df[oid_key].isin(oids_g)][[oid_key, ra_key, dec_key]]
    # oid_df_r = df[df[oid_key].isin(oids_r)][[oid_key, ra_key, dec_key]]
    # oids = {'g': oids_g, 'r': oids_r}
    # oid_df = {'g': oid_df_g, 'r': oid_df_r}

  def get_magnitude(self, ADU, zp, T):
    ADU = np.clip(ADU, 1, np.max(ADU))
    magnitude = zp - 2.5 * np.log10(ADU) - 0.2  # / T)
    return magnitude

  def get_magnitude_error(self, estimated_counts, estimated_count_error):
    estimated_counts = np.clip(estimated_counts, 1, np.max(estimated_counts))
    estimated_count_error = np.clip(estimated_count_error, 0,
                                    np.max(estimated_count_error))
    f = estimated_counts
    sigma_f = np.sqrt(estimated_count_error)
    sigma_m = 1.09 * (sigma_f / f)
    return sigma_m

  # this is counter intuituve due to lower magnitude is a brighter object
  def is_detected(self, magnitude, limmag5):
    return (magnitude - limmag5) < 0

  def check_lightcurve_lenght(self, field_data, lightcurve_indx,
      minimun_lenght=5):
    band_list = list(field_data[general_keys.LIGHTCURVES].keys())
    for band in band_list:
      estimated_counts = field_data[general_keys.ESTIMATED_COUNTS][band][
                           lightcurve_indx][:]
      if estimated_counts.shape[0] < minimun_lenght:
        return False
    return True

  def check_lightcurve_class(self, field_data, lightcurve_indx,
      classes_to_reject=[general_keys.ASTEROIDS,
                         general_keys.EMPTY_LIGHT_CURVE]):
    lc_type = field_data[general_keys.LC_TYPE][lightcurve_indx]
    if lc_type in classes_to_reject:
      return False
    return True

  def check_conditions_to_reject_light_curve(self, field_data, lightcurve_indx):
    conditions = [self.check_lightcurve_lenght(field_data, lightcurve_indx),
                  self.check_lightcurve_class(field_data, lightcurve_indx)]
    return all(conditions)

  def _remove_classes(self, label_names,
      classes_to_remove=[general_keys.ASTEROIDS,
                         general_keys.EMPTY_LIGHT_CURVE]):
    label_names = list(label_names)
    for class_to_remove in classes_to_remove:
      label_names.remove(class_to_remove)
    return np.array(label_names)

  def get_simulated_data_as_dict(self):
    sim_data = self.load_simulated_data()
    real_dataframe = self.load_dataframe()
    dict_sim_data = self._get_empty_dict_for_simulated_data()

    field_keys = list(sim_data.keys())
    # band_list = list(sim_data[field_keys[0]][general_keys.LIGHTCURVES].keys())
    print(field_keys)
    # print(band_list)
    dict_of_labels = {dataframe_keys.OID: [], 'label_name': [],
                      'label_value': []}
    label_names = np.unique(sim_data[field_keys[0]][general_keys.LC_TYPE])
    label_names = self._remove_classes(label_names)
    # print(label_names)

    oids_g = np.unique(real_dataframe[real_dataframe[dataframe_keys.FID] ==
                                      self.bands_to_num_dict[general_keys.G]][
                         dataframe_keys.OID])[
             :1000]
    oids_r = np.unique(real_dataframe[real_dataframe[dataframe_keys.FID] ==
                                      self.bands_to_num_dict[general_keys.R]][
                         dataframe_keys.OID])[
             :1000]
    oid_df_g = real_dataframe[real_dataframe[dataframe_keys.OID].isin(oids_g)][
      [dataframe_keys.OID, dataframe_keys.RA, dataframe_keys.DEC]]
    oid_df_r = real_dataframe[real_dataframe[dataframe_keys.OID].isin(oids_r)][
      [dataframe_keys.OID, dataframe_keys.RA, dataframe_keys.DEC]]
    oids = {'g': oids_g, 'r': oids_r}
    oid_df = {'g': oid_df_g, 'r': oid_df_r}
    object_id_indx = 0
    for field in field_keys:
      print('\n%s' % (field))
      field_data = sim_data[field]
      band_list = list(field_data[general_keys.LIGHTCURVES].keys())
      for lightcurve_indx in range(field_data[general_keys.LC_TYPE].shape[0]):
        if self.check_conditions_to_reject_light_curve(field_data,
                                                       lightcurve_indx) == False:
          continue

        oid = 'rod_est_2019_oid_%i' % object_id_indx
        lightcurve_name = field_data[general_keys.LC_TYPE][lightcurve_indx]
        lightcurve_value = np.argwhere(label_names == lightcurve_name)[0][0]
        dict_of_labels[dataframe_keys.OID].append(oid)
        dict_of_labels['label_name'].append(lightcurve_name)
        dict_of_labels['label_value'].append(lightcurve_value)

        for band in band_list:
          field_obs_cond = field_data[general_keys.OBS_COND]
          zp = field_obs_cond[general_keys.ZERO_POINT][band][:]
          exp_time = field_obs_cond[general_keys.EXP_TIME][band][:]
          limmag5 = field_obs_cond[general_keys.LIMMAG5][band][:]
          mjds = field_obs_cond[general_keys.OBS_DAY][band][:]

          lc_type = lightcurve_name
          estimated_counts = field_data[general_keys.ESTIMATED_COUNTS][band][
                               lightcurve_indx][:]
          estimated_error_counts = \
            field_data[general_keys.ESTIMATED_ERROR_COUNTS][band][
              lightcurve_indx][:]
          magnitude = self.get_magnitude(estimated_counts, zp, exp_time)
          magnitude_error = self.get_magnitude_error(estimated_counts,
                                                     estimated_error_counts)
          # write to dict that will be converted to df
          oid_list = ['rod_est_2019_oid_%i' % object_id_indx] * \
                     estimated_counts.shape[0]

          #
          random_oid = np.random.choice(oids[band].shape[0])
          random_oid_df = oid_df[band][
            oid_df[band][dataframe_keys.OID] == oids[band][random_oid]]
          ra_list = list(
              np.random.choice(random_oid_df[dataframe_keys.RA],
                               size=estimated_counts.shape[0]))
          dec_list = list(
              np.random.choice(random_oid_df[dataframe_keys.DEC],
                               size=estimated_counts.shape[0]))

          fid_list = [self.bands_to_num_dict[band]] * estimated_counts.shape[0]
          jd_list = list(mjds)
          magpsf_corr_list = list(magnitude)
          sigmapsf_corr_list = list(magnitude_error)
          lc_type_list = [lc_type] * estimated_counts.shape[0]
          detected_list = list(self.is_detected(magnitude, limmag5))

          # print('underliying magnitude: ', field_data[lightcurves_key][band][lightcurve_indx])
          # print('estimated counts: ', estimated_counts)
          # print('recovered magnitude: ', magnitude)
          # print('limmag5: ', limmag5)
          # print('detected: ', detected_list)

          # TODO: write label (lc_type) to file
          dict_sim_data[dataframe_keys.OID] += oid_list
          dict_sim_data[dataframe_keys.RA] += ra_list
          dict_sim_data[dataframe_keys.DEC] += dec_list
          dict_sim_data[dataframe_keys.JD] += jd_list
          dict_sim_data[dataframe_keys.FID] += fid_list
          dict_sim_data[dataframe_keys.MAGPSF_CORR] += magpsf_corr_list
          dict_sim_data[dataframe_keys.SIGMAPSF_CORR] += sigmapsf_corr_list
          dict_sim_data[dataframe_keys.LC_TYPE] += lc_type_list
          dict_sim_data[dataframe_keys.DETECTED] += detected_list
        object_id_indx += 1
    return dict_sim_data, dict_of_labels

  def get_simulated_data_as_df(self):
    dict_to_be_df, labels_dict = self.get_simulated_data_as_dict()
    sim_data_df_with_non_det = pd.DataFrame(dict_to_be_df)
    return sim_data_df_with_non_det, labels_dict

  def filter_non_detected_from_df(self, df):
    return df[df[dataframe_keys.DETECTED] == 1]

  def remove_unnecesary_colums(self, df):
    return df.drop([dataframe_keys.LC_TYPE, dataframe_keys.DETECTED], 1)


if __name__ == "__main__":
  ztf_light_curves_folder = os.path.join(PATH_TO_PROJECT, '..', 'datasets',
                                         'ZTF')
  sim_data_folder = os.path.join(PATH_TO_PROJECT, '..', 'datasets',
                                 'simulated_data', 'image_sequences')
  df_file_path = os.path.join(ztf_light_curves_folder,
                              'alerts_corr_10detections.pickle')  # 'alerts_corr_small.pickle')
  sim_data_file_path = os.path.join(sim_data_folder,
                                    'for_nacho_for_nacho1200_sigma5.hdf5')  # 'good_zero_points_good_zero_points10.hdf5')

  data_transformer = SimulatedData2Dataframe(
      path_simulated_data=sim_data_file_path, path_real_dataframe=df_file_path)
  sim_data_df_with_non_det, labels_dict = data_transformer.get_simulated_data_as_df()
  sim_data_df = data_transformer.filter_non_detected_from_df(
      sim_data_df_with_non_det)
  sim_data_df_droped = data_transformer.remove_unnecesary_colums(sim_data_df)

  sim_data_df_save_path = os.path.join(sim_data_folder,
                                       'sim_data_df_for_nacho.pkl')
  sim_data_labels_save_path = os.path.join(sim_data_folder,
                                           'sim_data_labels_for_nacho.pkl')
  sim_data_df_droped.to_pickle(sim_data_df_save_path)
  save_pickle(data=labels_dict, path=sim_data_labels_save_path)

  for i in range(10):
    print('\n %i' % i)
    for key in labels_dict.keys():
      print('%s %s' % (key, str(labels_dict[key][i])))

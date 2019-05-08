import os
import sys
import numpy as np
import pandas as pd
import h5py

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)


def get_magnitude(ADU, zp, T):
  ADU = np.clip(ADU, 1, np.max(ADU))
  magnitude = zp - 2.5 * np.log10(ADU) - 0.3  # / T)
  return magnitude


def get_magnitude_error(estimated_counts, estimated_count_error):
  estimated_counts = np.clip(estimated_counts, 1, np.max(estimated_counts))
  estimated_count_error = np.clip(estimated_count_error, 0,
                                  np.max(estimated_count_error))
  f = estimated_counts
  sigma_f = np.sqrt(estimated_count_error)
  sigma_m = 1.09 * (sigma_f / f)
  return sigma_m


# this is counter intuituve due to lower magnitude is a brighter object
def is_detected(magnitude, limmag5):
  return (magnitude - limmag5) < 0


class SimulatedData2Dataframe(object):

  def __init__(self, object_id_key='oid', filter_id_key='fid'):
    self.object_id_key = object_id_key
    self.filter_id_key = filter_id_key


estimated_counts_key = 'estimated_counts'
estimated_error_counts_key = 'estimated_error_counts'
g_key = 'g'
r_key = 'r'
lightcurves_key = 'lightcurves'
obs_cond_key = 'obs_cond'
obs_day_key = 'obs_day'
zero_point_key = 'zero_point'
exp_time_key = 'exp_time'
limmag5_key = 'limmag5'

oid_key = 'oid'
ra_key = 'ra'
dec_key = 'dec'
fid_key = 'fid'
jd_key = 'jd'
magpsf_corr_key = 'magpsf_corr'
sigmapsf_corr_key = 'sigmapsf_corr'
lc_type_key = 'lc_type'
detected_key = 'detected'

bands_to_num_dict = {'g': 1, 'r': 2}

if __name__ == "__main__":
  df_file_path = os.path.join(PATH_TO_PROJECT, '..', 'datasets', 'ZTF',
                              'alerts_corr_10detections.pickle')
  df = pd.read_pickle(df_file_path)
  sim_data_file_path = os.path.join(PATH_TO_PROJECT, '..', 'datasets',
                                    'simulated_data', 'image_sequences',
                                    'good_zero_points_good_zero_points10.hdf5')
  sim_data = h5py.File(sim_data_file_path, "r")

  dict_to_be_df = {oid_key: [], ra_key: [], dec_key: [], fid_key: [],
                   jd_key: [], magpsf_corr_key: [], sigmapsf_corr_key: [],
                   lc_type_key: [], detected_key: []}

  field_keys = sim_data.keys()
  oids_g = np.unique(df[df[fid_key] == bands_to_num_dict[g_key]][oid_key])[
           :1000]
  oids_r = np.unique(df[df[fid_key] == bands_to_num_dict[r_key]][oid_key])[
           :1000]
  oid_df_g = df[df[oid_key].isin(oids_g)][[oid_key, ra_key, dec_key]]
  oid_df_r = df[df[oid_key].isin(oids_r)][[oid_key, ra_key, dec_key]]
  oids = {'g': oids_g, 'r': oids_r}
  oid_df = {'g': oid_df_g, 'r': oid_df_r}
  object_id_indx = 0
  for field in field_keys:
    print('\n%s' % (field))
    field_data = sim_data[field]
    band_list = list(field_data[lightcurves_key].keys())
    for lightcurve_indx in range(field_data[lc_type_key].shape[0]):
      for band in band_list:
        zp = field_data[obs_cond_key][zero_point_key][band][:]
        exp_time = field_data[obs_cond_key][exp_time_key][band][:]
        limmag5 = field_data[obs_cond_key][limmag5_key][band][:]
        mjds = field_data[obs_cond_key][obs_day_key][band][:]

        lc_type = field_data[lc_type_key][lightcurve_indx]
        # print('\n%s %s %s %s' % (field, band, str(lightcurve_indx), lc_type))
        estimated_counts = field_data[estimated_counts_key][band][
                             lightcurve_indx][:]
        estimated_error_counts = field_data[estimated_error_counts_key][band][
                                   lightcurve_indx][:]
        magnitude = get_magnitude(estimated_counts, zp, exp_time)
        magnitude_error = get_magnitude_error(estimated_counts,
                                              estimated_error_counts)
        # write to dict that will be converted to df
        oid_list = ['rod_est_2019_oid_%i' % object_id_indx] * \
                   estimated_counts.shape[0]
        #
        random_oid = np.random.choice(oids[band].shape[0])
        random_oid_df = oid_df[band][
          oid_df[band][oid_key] == oids[band][random_oid]]
        ra_list = list(
            np.random.choice(random_oid_df[ra_key],
                             size=estimated_counts.shape[0]))
        dec_list = list(
            np.random.choice(random_oid_df[dec_key],
                             size=estimated_counts.shape[0]))
        fid_list = [bands_to_num_dict[band]] * estimated_counts.shape[0]
        jd_list = list(mjds)
        magpsf_corr_list = list(magnitude)
        sigmapsf_corr_list = list(magnitude_error)
        lc_type_list = [lc_type] * estimated_counts.shape[0]
        detected_list = list(is_detected(magnitude, limmag5))

        # print('underliying magnitude: ', field_data[lightcurves_key][band][lightcurve_indx])
        # print('estimated counts: ', estimated_counts)
        # print('recovered magnitude: ', magnitude)
        # print('limmag5: ', limmag5)
        # print('detected: ', detected_list)

        # TODO: write label (lc_type) to file
        dict_to_be_df[oid_key] += oid_list
        dict_to_be_df[ra_key] += ra_list
        dict_to_be_df[dec_key] += dec_list
        dict_to_be_df[jd_key] += jd_list
        dict_to_be_df[fid_key] += fid_list
        dict_to_be_df[magpsf_corr_key] += magpsf_corr_list
        dict_to_be_df[sigmapsf_corr_key] += sigmapsf_corr_list
        dict_to_be_df[lc_type_key] += lc_type_list
        dict_to_be_df[detected_key] += detected_list
      object_id_indx += 1

  for key in dict_to_be_df.keys():
    print('%s %s' % (key, str(len(dict_to_be_df[key]))))

  # dict to data frame
  sim_data_df_with_non_det = pd.DataFrame(dict_to_be_df)
  sim_data_df = sim_data_df_with_non_det[
    sim_data_df_with_non_det[detected_key] == 1]

  label_names = np.unique(sim_data_df[lc_type_key])


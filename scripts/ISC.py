import os, h5py, math, re
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_1samp
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle as rect
import matplotlib.patches as mpatches
from create_rawData import RawDataCreation
from preprocess import PreProc_missingValues


# Global\General Function
def plot_chunk_size_Vs_relevant_movies_by_threshold(min_chunk_size=0.7, max_chunk_size=1.6, step=0.15, threshold=0.2, facialExp="head", features=None, features_name=None):
    """
    min_chunk_size=0.7, max_chunk_size=1.6, step=0.15, threshold=0.3, facialExp="head"
        dict_chunk_size_num_movies = {0.7: 1, 0.85: 2, 1.0: 3, 1.15: 2, 1.3: 2, 1.5: 2, 1.6: 0}
    """
    movies_list = PreProc_missingValues().mIDs
    chunk_size_range = np.arange(min_chunk_size, max_chunk_size, step)
    dict_chunk_size_num_movies = {}
    num_relevant_movies_list = []
    if features == None:
        features_list = ISC.features_dict[facialExp]
    else:
        if features_name == None:
            print("argumment 'features_name' cant be None if argumnet 'features' is not None")
            return
        features_list = features
    ISC_instance = None
    for chunk_size in chunk_size_range:
        chunk_size = np.round(chunk_size, 1)
        print(chunk_size)
        num_relevant_movies = 0
        ISC_instance = None
        for feature in features_list:
            print(feature)
            for movie in movies_list:
                print(movie)
                ISC_instance = ISC(movie=movie, feature=feature, facialExp=facialExp)
                ISC_data = ISC_instance.get_chunked_ISC(chunk_size)
                bars = ISC_data['average_correlation']
                max_ISC = np.max(bars)
                if max_ISC >= threshold:
                    num_relevant_movies += 1
        dict_chunk_size_num_movies[chunk_size] = num_relevant_movies
        num_relevant_movies_list.append(num_relevant_movies)
    # dict_chunk_size_num_movies = {0.7: 1, 0.85: 2, 1.0: 3, 1.15: 2, 1.3: 2, 1.5: 2, 1.6: 0}
    # num_relevant_movies_list = [1,2,3,2,2,2,0]
    print("plot chunk size")
    print(dict_chunk_size_num_movies)
    print(chunk_size_range)
    print(num_relevant_movies_list)
    plt.figure()
    plt.plot(chunk_size_range, num_relevant_movies_list, '-ok')
    plt.rcParams.update({'font.size': 10})
    title = "{}: Chunk size Vs number of movies that exceed the threshold {}".format(ISC_instance.facialExp, threshold)
    plt.title(title)
    plt.xticks(chunk_size_range)
    plt.ylabel("number of movies")
    plt.xlabel("chunk size")
    plt.grid(color='lightgray')
    fig_name = "{} Chunk size Vs number of movies that exceed the threshold {}".format(ISC_instance.facialExp,
                                                                                       threshold) + str(
        features_name) + ".png"
    plt.savefig(fig_name)
    plt.show()

def plot_threshold_Vs_relevant_movies(min_threshold=0.3, max__threshold=0.4, chunk_size=1.65, step=0.04, facialExp="head", features=None, features_name=None):
    # movie="m1"
    # facialExp="head"
    # p=ISC(movie=movie, feature="Yaw", facialExp=facialExp)
    movies_list = PreProc_missingValues().mIDs
    threshold_range = np.arange(min_threshold, max__threshold, step)
    dict_threshold_num_movies = {}
    num_relevant_movies_list = []
    if features == None:
        features_list = ISC.features_dict[facialExp]
    else:
        if features_name == None:
            print("argumment 'features_name' cant be None if argumnet 'features' is not None")
            return
        features_list = features
    ISC_instance = None
    for threshold in threshold_range:
        threshold = np.round(threshold, 2)
        print(threshold)
        num_relevant_movies = 0
        for feature in features_list:
            print(feature)
            for movie in movies_list:
                print(movie)
                ISC_instance = ISC(movie=movie, feature=feature, facialExp=facialExp)
                ISC_data = ISC_instance.get_chunked_ISC(chunk_size)
                bars = ISC_data['average_correlation']
                max_ISC = np.max(bars)
                if max_ISC >= threshold:
                    num_relevant_movies += 1
        dict_threshold_num_movies[threshold] = num_relevant_movies
        num_relevant_movies_list.append(num_relevant_movies)
    plt.figure()
    plt.plot(threshold_range, num_relevant_movies_list, '-ok')
    plt.rcParams.update({'font.size': 10})
    title = "{}: Threshold Vs number of movies with chunk size of {} seconds".format(ISC_instance.facialExp, chunk_size)
    plt.title(title)
    plt.xticks(threshold_range)
    plt.ylabel("number of movies")
    plt.xlabel("cthreshold")
    plt.grid(color='lightgray')
    plt.savefig("{} Threshold Vs number of movies with chunk size of {} seconds".format(ISC_instance.facialExp,
                                                                                        chunk_size) + features_name + ".png")
    plt.show()

def get_sub_group_by_conditions(feature="Yaw", corr_threshold=0.4, appearance_threshold=3):
    movies_list = ["m1", "m3", "m4", "m5", "m6", "m7", "m10", "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19"]
    mean_corrs_above_threshold_movie_dict = {}
    for m in movies_list:
        p = ISC(movie=m, feature=feature, facialExp="head", is_external_group=False)
        corrs = p.subjects_data_corrs()
        mean_corrs = p.subject_stats(corrs, np.mean)
        mean_corrs_above_threshold = mean_corrs[mean_corrs > corr_threshold]
        mean_corrs_above_threshold_movie_dict[m] = mean_corrs_above_threshold
    subjects = []
    for k, v in mean_corrs_above_threshold_movie_dict.items():
        if v.shape[0] > 0:
            subjects += list(v.index)
    series_subjects = pd.Series(subjects, dtype="float64")
    np.unique(series_subjects).size
    subjects_counts = series_subjects.value_counts()
    subjects_counts_above_3 = subjects_counts[subjects_counts > appearance_threshold]
    relevant_subjects = np.unique(subjects_counts_above_3.index)
    series_relevant_subjects = pd.Series(relevant_subjects, dtype="float64")
    series_relevant_subjects.to_csv("relevant_subjects_{}.csv".format(feature), index=False, header=False)
    return series_relevant_subjects


class Movie:

    class Events:
        hazardPressesQualtrics_dir = os.path.join(RawDataCreation.datadir, "hazardPressesQualtrics.xlsx")
        independent_hazardPressesQualtrics_dir = os.path.join(RawDataCreation.datadir, "independentHazardPressesQualtrics.xlsx")
        threshold_events = 0.14
        threshold_events_dependent = 0.18

        def __init__(self, mid, fps=30):
            """
            The mid is "m8", "m7" etc.
            """
            self.mid = mid
            self.fps = fps
            self.press_data = pd.read_excel(Movie.Events.hazardPressesQualtrics_dir, sheet_name=self.mid)
            self.independent_press_data = pd.read_excel(Movie.Events.independent_hazardPressesQualtrics_dir,
                                                        sheet_name=self.mid)

        def event_times(self):
            # e=p.movie.Events("m1")
            events_data = self.press_data[(self.press_data["Movie"].str.contains("press time")) | (
                self.press_data["Movie"].str.contains("Percentage of"))]
            events_data = events_data[events_data.filter(like='Hazard').columns]  # keep hazards cols
            events_data = events_data[events_data >= Movie.Events.threshold_events].dropna(axis=1)
            events_data = events_data.iloc[:-1, ]
            # events_data_diff = events_data.diff()[1:]
            # keep_events_name = events_data.columns[(events_data_diff != 0).any(axis=0)]
            # events_data = events_data[keep_events_name]
            return events_data

        def get_num_hazards_road_experience_by_participant(self, subject, is_dependent=True):
            # subject = "200"
            # e = p.movie.Events(p.movie.id)
            if is_dependent:
                # events_identification = self.press_data[self.press_data["Movie"].str.contains("Percentage of")]
                # events_identification = events_identification[events_identification.filter(like='Hazard').columns]  # keep hazards cols
                # filtered_events_identification = events_identification[events_identification >= Movie.Events.threshold_events].dropna(axis=1)
                events_data = self.press_data[["Participant"] + list(self.press_data.filter(like='Hazard').columns)]
                subject_hazards = events_data[events_data["Participant"] == int(subject)].dropna(axis=1)
                num_hazard_identify = subject_hazards[subject_hazards.filter(like='Hazard').columns].shape[
                    1]  # keep hazards cols
                subject_row_index = \
                self.press_data[self.press_data["Participant"] == int(subject)].index.values.astype(int)[0]
                road_experience = self.press_data["The number of times they have been this way"].iloc[subject_row_index]
                if np.isnan(road_experience):
                    road_experience = 0
                return num_hazard_identify, road_experience
            else:
                events_identification = self.independent_press_data[
                    self.independent_press_data["Movie"].str.contains("origin")]
                events_identification = events_identification[
                    events_identification.filter(like='Hazard').columns]  # keep hazards cols
                filtered_events_identification = events_identification[
                    events_identification >= Movie.Events.threshold_events_dependent].dropna(axis=1)
                events_data = self.independent_press_data[
                    ["Participant"] + list(filtered_events_identification.columns)]
                subject_hazards = events_data[events_data["Participant"] == int(subject)].dropna(axis=1)
                num_hazard_identify = subject_hazards[subject_hazards.filter(like='Hazard').columns].shape[
                    1]  # keep hazards cols
                subject_row_index = self.independent_press_data[
                    self.independent_press_data["Participant"] == int(subject)].index.values.astype(int)[0]
                road_experience = self.independent_press_data["The number of times they have been this way"].iloc[
                    subject_row_index]
                if np.isnan(road_experience):
                    road_experience = 0
                return num_hazard_identify, road_experience

    def __init__(self, movie, facialExp="head", data_type="processed_data_no_NaN"):
        """
        The movie id, movie, is "m8", "m7" etc.
        """
        self.id = movie
        self.movie_number = int(re.sub("[^\d]", "", self.id))
        self.facialExp = facialExp
        self.data_type = data_type  # there are some types of data in hdf5
        self.subjects_data_dict = dict()
        self.db_loaded = False
        self.meta = None
        self.load_db()
        assert self.db_loaded, "can't load the data aborting everything"
        self.data_length = self.meta['length']
        self.fps = self.meta['fps']
        self.events = Movie.Events(mid=self.id, fps=self.fps)

    def get_subjects(self):
        return self._subjects

    def load_db(self):
        hd5dir = os.path.join(RawDataCreation.datadir, "{}_DB.hd5".format(self.facialExp))
        hd = h5py.File(hd5dir, "r")
        m_db = hd[self.id]  # get data base of current movie
        self.meta = dict(m_db.attrs)
        self.features = self.meta["features"].tolist()
        m_data_db = m_db[self.data_type]
        subjs = list()
        for s_id in m_data_db:
            self.subjects_data_dict[s_id] = pd.DataFrame(m_data_db[s_id][()], columns=self.features)
            subjs.append(s_id)
        self._subjects = subjs
        hd.close()
        self.db_loaded = True


class ISC:
    opencv_palette = {
        'red': "r",
        'blue': "b",
        'yellow': "y",
        'magenta': "m",
        'cyan': "c",
        'green': "g"
    }
    dict_colors_featurs = {
        'Yaw': "yellow", 'Pitch': "blue", 'Roll': "darkorange", 'Brow Furrow': "black", 'Brow Raise': "grey",
        'Lip Corner Depressor': "darkred",
        'InnerBrowRaise': "red",
        'EyeClosure': "tomato", 'NoseWrinkle': "chocolate", 'UpperLipRaise': "sienna", 'LipSuck': "darkorange",
        'LipPress': "gold", 'MouthOpen': "yellow",
        'ChinRaise': "yellowgreen", 'Smirk': "black", 'LipPucker': "red", 'Cheek Raise': "chocolate", 'Dimpler': "cyan",
        'Eye Widen': "darkred",
        'Lid Tighten': "blue",
        'Lip Stretch': "mediumpurple", 'Jaw Drop': "mediumorchid", 'Anger': "violet", 'Sadness': "purple",
        'Disgust': "fuchsia", 'Joy': "deeppink", 'Surprise': "pink",
        'Fear': "darkred", 'Contempt': "darkkhaki"
    }
    features_dict = PreProc_missingValues.chosenCols
    plotwidth = 14
    plotheight = 6
    movies_ID = PreProc_missingValues().mIDs
    AU_1 = ['Brow Furrow', 'Brow Raise', 'InnerBrowRaise', 'EyeClosure', 'Eye Widen']  # Action Units group 1
    AU_2_1 = ['MouthOpen', 'Smirk', 'Dimpler', 'Lid Tighten', 'Jaw Drop']  # Action Units group 2.1
    AU_2_2 = ['Lip Corner Depressor', 'UpperLipRaise', 'LipSuck', 'LipPress', 'LipPucker',
              'Lip Stretch']  # Action Units group 2.2
    AU_3 = ['ChinRaise', 'Cheek Raise', 'NoseWrinkle']  # Action Units group 3
    EM = PreProc_missingValues.EM  # emotions
    facial = AU_1 + AU_2_1 + AU_2_2 + AU_3 + EM
    AUs = AU_1 + AU_2_1 + AU_2_2 + AU_3
    datadir = PreProc_missingValues.datadir
    external_group = pd.read_csv(os.path.join(datadir, 'external_group.csv')).astype(
        str).squeeze().values  # included 5 years actual experience
    movies_turnings_data = pd.read_csv(os.path.join(datadir, 'movies_turnings.csv'))
    turnings_map_dir = os.path.join(RawDataCreation.datadir, "turnings_map.xlsx")

    def __init__(self, movie="m1", feature="Yaw", facialExp="head", is_mean_all_features=False, is_external_group=False):
        self.movie = Movie(movie=movie, facialExp=facialExp) if isinstance(movie, str) else movie
        self.subjects = np.array(self.movie.get_subjects())
        self.feature = feature
        self.facialExp = facialExp
        self.group_color = 'b'  # blue
        self.group_name = "experienced"
        self.is_mean_all_features = is_mean_all_features
        self.is_external_group = is_external_group

    def get_processed_data_by_subjid(self, subjid):
        if self.movie.db_loaded:
            if self.is_external_group:
                if subjid not in self.subjects:
                    return None
            if self.is_mean_all_features:
                processed_data = self.movie.subjects_data_dict[subjid]
                MediaTime = processed_data["MediaTime"].values
                mean = np.mean(processed_data[ISC.features_dict[self.facialExp]].values, axis=1)
                mean_processed_data = np.column_stack((MediaTime, mean))
                mean_processed_data = pd.DataFrame(mean_processed_data, columns=["MediaTime", "Mean"])
                return mean_processed_data
            else:
                return self.movie.subjects_data_dict[subjid]
        print("There is no data available for the subject {} and movie {}".format(subjid, self.movie.id))
        return None

    def correlation_calculation(self, subject1, subject2, slot_start, slot_end):
        subject1_data = self.get_processed_data_by_subjid(subject1)
        subject2_data = self.get_processed_data_by_subjid(subject2)
        if (subject1_data is None) | ((subject2_data is None)):
            return None
        if self.is_mean_all_features:
            subject1_data = subject1_data["Mean"]
            subject2_data = subject2_data["Mean"]
        else:
            subject1_data = subject1_data[self.feature]
            subject2_data = subject2_data[self.feature]
        chunked_subject1_data = subject1_data[slot_start: slot_end]
        chunked_subject2_data = subject2_data[slot_start:slot_end]
        corr = np.corrcoef(chunked_subject1_data, chunked_subject2_data)[0, 1]
        return corr

    def get_correlation_data(self, subjects, slot_start, slot_end):
        if self.is_external_group:
            subjects = ISC.external_group
        all_data_couples = list(combinations(subjects, 2))
        corr_data = pd.Series(index=pd.MultiIndex.from_tuples(all_data_couples, names=['subject1', 'subject2']), dtype='float64')
        for subject1, subject2 in corr_data.index:
            # subject1, subject2 = corr_data.index[0]
            corr = self.correlation_calculation(subject1, subject2, slot_start, slot_end)
            corr_data[subject1, subject2] = corr
        # return corr_data[corr_data != 5]
        return corr_data.dropna()

    def get_chunked_ISC(self, chunk_size=1):
        """
        chunk_size: numeric length in seconds of parts
        """
        ISC_data = np.empty((0, 4))  # cols names: ['average_correlation', 'std_correlation', 't_statistic', 'p_value']
        done = False
        step = int(chunk_size * self.movie.fps)  # the size\length of each bar of the graph
        current_slot_start = 0  # index number to start
        current_slot_end = step  # index number to end
        if self.is_external_group:
            subjects_list = ISC.external_group
        else:
            subjects_list = self.subjects
        while not done:
            current_pearson_corrs_data = self.get_correlation_data(subjects=subjects_list,
                                                                   slot_start=current_slot_start,
                                                                   slot_end=current_slot_end)
            ISC_data = np.vstack(
                (ISC_data, (round(current_pearson_corrs_data.mean(), 3), round(current_pearson_corrs_data.std(), 3),
                            round(ttest_1samp(current_pearson_corrs_data, 0, nan_policy='omit').statistic, 3),
                            round(ttest_1samp(current_pearson_corrs_data, 0, nan_policy='omit').pvalue, 3))))
            if self.movie.data_length - current_slot_end > step:
                current_slot_start = current_slot_end  # index number to start
                current_slot_end = current_slot_start + step  # index number to end
            else:
                done = True
        return pd.DataFrame(ISC_data, columns=['average_correlation', 'std_correlation', 't_statistic', 'p_value'],
                            dtype='float64')

    def get_all_subjetcs_feature_data(self, feature):
        data_length = self.movie.data_length
        if self.is_external_group:
            subjects_list = ISC.external_group
        else:
            subjects_list = self.subjects
        all_subjects_feature_data = pd.DataFrame(index=range(int(data_length)), columns=subjects_list)
        for s in subjects_list:
            data = self.get_processed_data_by_subjid(s)
            if data is None:
                continue
            if self.is_mean_all_features:
                all_subjects_feature_data[s] = data["Mean"]
            else:
                all_subjects_feature_data[s] = data[feature]
        return all_subjects_feature_data

    def reset_media_time_col(self, data):
        data_diff = data.diff()
        mean_data_diff = np.round(np.mean(data_diff[1:]))
        data_diff[0] = mean_data_diff
        newdata = data_diff.cumsum(axis=0)
        return newdata

    def plot_mean_feature_by_time(self, ax=None, xticks=None, ticks=None):
        # ax = axs[1]
        # xticks = bars_time_seconds
        # ticks = ticks
        all_subjetcs_feature_data = self.get_all_subjetcs_feature_data(self.feature).dropna(axis=1)
        mean = all_subjetcs_feature_data.mean(axis=1).values
        std = all_subjetcs_feature_data.std(axis=1).values
        show = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(ISC.plotwidth, ISC.plotheight))
            title = "{:s}: feature {:s} by time of movie {:s}".format(self.facialExp, self.feature, self.movie.id)
            plt.title(title)
            show = True
        label = "road hazard"
        events_data = self.movie.events.event_times()
        for col in range(0, events_data.shape[1]):
            ax.axvspan(xmin=events_data.iloc[0, col], xmax=events_data.iloc[1, col], alpha=0.3,
                       color=ISC.opencv_palette["green"], label=label)
            if col == 0:
                label = None
        media_time_movie = self.reset_media_time_col(
            self.get_processed_data_by_subjid(all_subjetcs_feature_data.columns[0])["MediaTime"])
        _, pltx = divmod(media_time_movie.values / 1000, 60)
        ax.plot(pltx, mean, lw=2, label='mean')
        ax.fill_between(pltx, mean, mean - std, color='lightgrey', label='std', clip_on=True)
        ax.fill_between(pltx, mean, mean + std, color='lightgrey')
        if xticks is not None:
            plt.xticks(np.round(xticks, 1))
        if ticks is not None:
            ax.set_xlim(0, ticks)
        ax.set_ylabel("Mean of feature {:s}".format(self.feature))
        ax.set_xlabel("seconds")
        ax.grid()
        ax.legend()
        if self.facialExp == 'facial':
            plt.ylim(bottom=0)
        if show:
            plt.savefig("feature {:s} by time of movie {:s}".format(self.feature, self.movie.id) + ".png")
            plt.show()

    def plot_chunked(self, chunk_size=1, add_events=True, threshold=None, alpha=0.05):
        # movie="m1"
        # facialExp="head"
        # chunk_size=1
        # add_events = True
        # threshold = None
        # alpha = 0.05
        # p=ISC(is_external_group=True)
        # p.plot_chunked()
        ISC_data = self.get_chunked_ISC(chunk_size)
        bars = ISC_data['average_correlation']
        if threshold != None:
            max_ISC = np.max(bars)
            if max_ISC < threshold:
                print("The max value of ISC for movie {:s} does not exceed the set threshold value {}".format(
                    self.movie.id, threshold))
                return None
        errs = ISC_data['std_correlation'].values
        ticks = math.ceil(self.movie.data_length / self.movie.fps)
        bars_time_seconds = [chunk_size * i for i in range(bars.shape[0] + 1)]
        bars_middle_time_seconds = np.array(bars_time_seconds[0:-1]) + np.diff(bars_time_seconds) / 2
        fig, axs = plt.subplots(2, 1, figsize=(ISC.plotwidth * 1.2, round(ISC.plotheight * 1.5)))
        ax = axs[0]
        self.plot_mean_feature_by_time(ax=axs[1], xticks=bars_time_seconds, ticks=ticks)
        if add_events:
            events_data = self.movie.events.event_times()
            label = "road hazard"
            for col in range(0, events_data.shape[1]):
                ax.axvspan(xmin=events_data.iloc[0, col], xmax=events_data.iloc[1, col], alpha=0.3,
                           color=ISC.opencv_palette["green"], label=label)
                if col == 0:
                    label = None
            # ax.add_artist(ax.legend(loc=(0.85, 0.8)))

        for j in range(bars.shape[0]):  # range of all the bars and for each bar checking sig (and corr. if necessary)
            pValue = "%.2f" % round(ISC_data.loc[j]['p_value'], 2)
            corr_j = ISC_data.loc[j]['average_correlation']
            if float(pValue) < alpha:  # apply bar (and text) color by corrs
                colorBar = 'firebrick'
            else:
                colorBar = 'darkblue'
            ax.add_patch(rect((chunk_size * j, 0), chunk_size, corr_j, facecolor=colorBar, edgecolor='k'))
        notSigColor = mpatches.Patch(color='darkblue', label='Not sig.')
        sigColor = mpatches.Patch(color='firebrick', label='Sig.')
        hazard_color = mpatches.Patch(color=ISC.opencv_palette["green"], label="road hazard", alpha=0.3)
        ax.legend(handles=[notSigColor, sigColor, hazard_color], loc='upper left')

        plt.rcParams.update({'font.size': 12})
        if self.is_mean_all_features:
            ylab = "ISC of mean {} features".format(self.facialExp)
            title = "{:s}: {:s} divided to {:.1f}s parts".format(self.facialExp, self.movie.id, float(chunk_size))
            ax.set_xlabel("Seconds")
        else:
            ylab = "ISC of feature {:s}".format(self.feature)
            title = "{:s}: {:s} divided to {:.1f}s parts of feature {:s}".format(self.facialExp, self.movie.id,
                                                                                 float(chunk_size), self.feature)
        plt.suptitle(title)
        ax.errorbar(bars_middle_time_seconds, bars.values, errs, linestyle="none", color="darkgray", capsize=3,
                    label="")
        ax.set_ylim(min((bars.values - errs / 2).min(), 0) - 0.3, (bars.values + errs / 2).max() + 0.3)
        ax.set_xticks(np.round(bars_time_seconds, 1))
        ax.set_ylabel(ylab)
        ax.set_xlim(0, ticks)
        ax.grid(color='lightgray')
        if self.is_mean_all_features:
            plt.savefig(ylab + "{}_.png".format(self.movie.id))
        else:
            plt.savefig("{} of {} of {} and chunk size is {}.png".format(self.feature, self.facialExp, self.movie.id,
                                                                         chunk_size))
        plt.show()

    def plot_chunked_diff_features_plots_one_fig(self, chunk_size=1, features=None, threshold=None, features_name=None, alpha=0.05):
        # movie="m1"
        # facialExp="head"
        # features=ISC.features_dict[facialExp]
        # p=ISC(movie=movie, feature="Yaw", facialExp=facialExp)
        if features == None:
            features_list = self.features_dict[self.facialExp]
        else:
            if features_name == None:
                print("argumment 'features_name' cant be None if argumnet 'features' is not None")
                return
            features_list = features
        plt.rcParams.update({'font.size': 24})
        dict_feature_ISC = {}
        max_ISC = -math.inf
        min_ISC = math.inf
        for feature in features_list:
            print(feature)
            ISC_temp_instance = ISC(movie=self.movie.id, feature=feature, facialExp=self.facialExp,
                                    is_mean_all_features=self.is_mean_all_features, is_external_group=self.is_external_group)
            ISC_data = ISC_temp_instance.get_chunked_ISC(chunk_size)
            bars = ISC_data['average_correlation']
            if threshold != None:
                max_current_ISC = np.max(bars)
                min_current_ISC = np.min(bars)
                if max_current_ISC > max_ISC:
                    max_ISC = max_current_ISC
                if min_current_ISC < min_ISC:
                    min_ISC = min_current_ISC
                    if max_current_ISC < threshold:
                        continue
            dict_feature_ISC[feature] = ISC_data
        ticks = math.ceil(self.movie.data_length / self.movie.fps)
        bars_time_seconds = [chunk_size * i for i in range(bars.shape[0] + 1)]
        bars_middle_time_seconds = np.array(bars_time_seconds[0:-1]) + np.diff(bars_time_seconds) / 2
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(ISC.plotwidth * 2.5, ISC.plotheight * len(features_list)))
        events_data = self.movie.events.event_times()
        label = "road hazard"
        legends = {}
        legends["sig"] = mpatches.Patch(color='firebrick', label="Sig.")
        legends["hazards"] = mpatches.Patch(color=ISC.opencv_palette["green"], label="road hazard", alpha=0.3)
        title = "ISC of movie {} on {} features, chunk size is {} seconds".format(self.movie.id, self.facialExp, chunk_size)
        for i, ax in enumerate(axs):
            # i=0
            # ax = axs[i]
            feature = features_list[i]
            ISC_data = dict_feature_ISC[feature]
            for col in range(0, events_data.shape[1]):
                ax.axvspan(xmin=events_data.iloc[0, col], xmax=events_data.iloc[1, col], alpha=0.3, color=ISC.opencv_palette["green"], label=label)
                if col == 0:
                    label = None
            for j in range(
                    bars.shape[0]):  # range of all the bars and for each bar checking sig (and corr. if necessary)
                # j=0
                pValue = "%.2f" % np.round(ISC_data['p_value'].loc[j], 2)
                corr_j = ISC_data['average_correlation'].loc[j]
                if float(pValue) < alpha:  # apply bar (and text) color by corrs
                    colorBar = 'firebrick'
                else:
                    colorBar = "b"
                ax.add_patch(rect((chunk_size * j, 0), chunk_size, corr_j, facecolor=colorBar, edgecolor='k'))
            errs = ISC_data['std_correlation'].values
            if i == 0:
                ax.set_title(title)
            ax.errorbar(bars_middle_time_seconds, bars.values, errs, linestyle="none", color="darkgray", capsize=3, label="")
            ax.set_ylim(min((bars.values - errs / 2).min(), 0) - 0.3, (bars.values + errs / 2).max() + 0.3)
            ax.set_xticks(np.round(bars_time_seconds, 1))
            ax.set_xlim(0, ticks)
            ax.set_ylabel("ISC of {}".format(feature))
            ax.grid(color='lightgray')
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.legend(handles=legends.values(), loc=(1.01, 2.9))
        plt.xlabel("Seconds")
        if features is None:
            plt.savefig(title + "_1fig_all.png")
        else:
            plt.savefig(title + features_name + ".png")
        plt.show()

    def plot_mean_by_time_diff_features_plots_one_fig(self, features=None, features_name=None, is_turnings=False, is_events=False):
        #p = ISC(movie="m4")
        if features == None:
            features_list = self.features_dict[self.facialExp]
        else:
            if features_name == None:
                print("argumment 'features_name' cant be None if argumnet 'features' is not None")
                return
            features_list = features
        plt.rcParams.update({'font.size': 24})
        fig, axs = plt.subplots(nrows=3, figsize=(ISC.plotwidth * 2, ISC.plotheight * len(features_list)))
        title = "{:s} features by time of movie {:s}".format(self.facialExp, self.movie.id)
        for i, ax in enumerate(axs):
            # i=0
            # ax = axs[i]
            # print(i)
            feature = features_list[i]
            all_subjetcs_feature_data = self.get_all_subjetcs_feature_data(feature)
            mean = all_subjetcs_feature_data.mean(axis=1).values
            std = all_subjetcs_feature_data.std(axis=1).values
            if is_turnings:
                label = "Turning"
                turnings_data = pd.read_excel(self.turnings_map_dir, sheet_name=self.movie.id)
                if turnings_data.shape[1] > 1:
                    for col in range(1, turnings_data.shape[1]):
                        ax.axvspan(xmin=turnings_data.iloc[0, col], xmax=turnings_data.iloc[1, col], alpha=0.3,
                                   color=ISC.opencv_palette["yellow"], label=label)
                        if col == 1:
                            label = None
            if is_events:
                label = "road hazard"
                events_data = self.movie.events.event_times()
                for col in range(0, events_data.shape[1]):
                    ax.axvspan(xmin=events_data.iloc[0, col], xmax=events_data.iloc[1, col], alpha=0.3,
                               color=ISC.opencv_palette["green"], label=label)
                    if col == 0:
                        label = None
            ISC_temp_instance = ISC(movie=self.movie.id, feature=feature, facialExp=self.facialExp)
            media_time_movie = ISC_temp_instance.reset_media_time_col(ISC_temp_instance.get_processed_data_by_subjid(all_subjetcs_feature_data.columns[0])["MediaTime"])
            _, pltx = divmod(media_time_movie.values / 1000, 60)
            ax.plot(pltx, mean, lw=2, label='mean')
            lower_std = mean - std
            upper_std = mean + std
            ax.fill_between(pltx, mean, lower_std, color='lightgrey', label='std', clip_on=True)
            ax.fill_between(pltx, mean, upper_std, color='lightgrey')
            ticks = math.ceil(self.movie.data_length / self.movie.fps)
            if i == 0:
                ax.set_title(title)
            ax.set_xlim(0, ticks)
            ax.set_yticks(np.arange(np.min(lower_std) - 1, np.max(upper_std) + 1, 2))
            ax.set_ylabel("Mean of {:s}".format(feature))
            ax.grid()
        plt.xlabel("seconds")
        ax.legend(loc=(1.005, 3))
        if features is None:
            plt.savefig(title + "_all.png")
        else:
            plt.savefig(title + features_name + ".png")
        plt.show()
        plt.show()

    def plot_chunked_all_feaures_bar_plot(self, chunk_size=1, features=None, threshold=None, features_name=None):
        # p.plot_chunked_all_feaures_bar_plo()
        # movie="m11"
        # facialExp="head"
        # features=ISC.features_dict[facialExp]
        # p=ISC(movie=movie, feature="Roll", facialExp=facialExp)
        if features == None:
            features_list = self.features_dict[self.facialExp]
        else:
            if features_name == None:
                print("argumment 'features_name' cant be None if argumnet 'features' is not None")
                return
            features_list = features
        dict_feature_ISC = {}
        max_ISC = -math.inf
        min_ISC = math.inf
        for feature in features_list:
            # feature = features_list[0]
            print(feature)
            ISC_temp_instance = ISC(movie=self.movie.id, feature=feature, facialExp=self.facialExp,
                                    is_mean_all_features=self.is_mean_all_features)
            ISC_data = ISC_temp_instance.get_chunked_ISC(chunk_size)
            f_data = ISC_data[['average_correlation', 'p_value']]
            max_current_ISC = np.max(f_data['average_correlation'])
            min_current_ISC = np.min(f_data['average_correlation'])
            if max_current_ISC > max_ISC:
                max_ISC = max_current_ISC
            if min_current_ISC < min_ISC:
                min_ISC = min_current_ISC
            if threshold != None:
                if max_current_ISC < threshold:
                    continue
            dict_feature_ISC[feature] = f_data
        ticks = math.ceil(self.movie.data_length / self.movie.fps)
        bars_time_seconds = [chunk_size * i for i in range(f_data.shape[0] + 1)]
        fig, ax = plt.subplots(figsize=(ISC.plotwidth, ISC.plotheight))
        # fig = plt.figure(figsize=(ISC.plotwidth, ISC.plotheight))
        events_data = self.movie.events.event_times()
        label = "road hazard"
        for col in range(0, events_data.shape[1]):
            plt.axvspan(xmin=events_data.iloc[0, col], xmax=events_data.iloc[1, col], alpha=0.3,
                        color=ISC.opencv_palette["green"], label=label)
            if col == 0:
                label = None
        legends = {}
        legends["sig"] = mpatches.Patch(color='firebrick', label="Sig.")
        legends["hazards"] = mpatches.Patch(color=ISC.opencv_palette["green"], label="road hazard")
        for i, feature in enumerate(features_list):
            # i=2
            # feature = features_list[i]
            print(feature)
            bars_f = dict_feature_ISC[feature]
            colorline = "gray"
            legends[feature] = mpatches.Patch(color=ISC.dict_colors_featurs[feature], label=feature)
            for j in range(
                    bars_f.shape[0]):  # range of all the bars and for each bar checking sig (and corr. if necessary)
                pValue = "%.2f" % round(bars_f.loc[j]['p_value'], 2)
                corr_j = bars_f.loc[j]['average_correlation']
                position = (chunk_size * j) + i / len(features_list)
                if float(pValue) < 0.05:  # apply bar (and text) color by corrs
                    colorBar = 'firebrick'
                else:
                    colorBar = ISC.dict_colors_featurs[feature]
                ax.add_patch(rect((position, 0), chunk_size * (1 / len(features_list)), corr_j, facecolor=colorBar,
                                  edgecolor=colorline, linewidth=1))
                if float(pValue) < 0.05:  # apply bar (and text) color by corrs
                    plt.plot(position + chunk_size * (1 / len(features_list) / 2), corr_j + 0.01, '*',
                             color=ISC.dict_colors_featurs[feature], markersize=9)
        plt.legend(handles=legends.values())
        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.rcParams.update({'font.size': 12})
        title = "ISC of movie {} on {} features, chunk size is {} seconds".format(self.movie.id, self.facialExp,
                                                                                  chunk_size)
        plt.title(title)
        plt.xticks(np.round(bars_time_seconds, 1))
        plt.ylabel("ISC")
        plt.xlabel("seconds")
        plt.xlim(0, ticks)
        plt.ylim(min_ISC - 0.1, max_ISC + 0.1)
        plt.grid(color='dimgray')
        if features == None:
            plt.savefig(title + "_bars_all.png")
        else:
            plt.savefig(title + features_name + "_bars.png")
        plt.show()

    def subjects_data_corrs(self):
        if self.is_external_group:
            subjects = ISC.external_group
        else:
            subjects = self.subjects
        tups = [(subjects[i], subjects[j]) for i in range(subjects.size - 1) for j in
                range(i + 1, subjects.size)]  # create all the cpossible couples of subjects
        corrs = pd.Series(index=pd.MultiIndex.from_tuples(tups, names=['subject1', 'subject2']), dtype="float64")
        for s1, s2 in corrs.index:
            # s1, s2 = corrs.index[0]
            subject1_data = self.get_processed_data_by_subjid(s1)[self.feature]
            subject2_data = self.get_processed_data_by_subjid(s2)[self.feature]
            corrs[s1, s2] = np.corrcoef(subject1_data, subject2_data)[0, 1]
        return corrs

    def subjects_data_corrs_by_chunk_size(self, chunk_size=1):
        all_ISC_data = {}
        step = int(chunk_size * self.movie.fps)  # the size\length of each bar of the graph
        slot_start = 0  # index number to start
        slot_end = step  # index number to end
        if self.is_external_group:
            subjects = ISC.external_group
        else:
            subjects = self.subjects
        for s in subjects:
            # s = subjects[0]
            subject_ISC_data = np.empty((0, 4))
            other_subjects = list(set(subjects) - set([s]))
            done = False
            while not done:
                chunked_corrs = []
                for s2 in other_subjects:
                    # s2 = other_subjects[0]
                    # s2 = other_subjects[0]
                    corr = self.correlation_calculation(s, s2, slot_start, slot_end)
                    chunked_corrs.append(corr)
                subject_ISC_data = np.vstack(
                    (subject_ISC_data, (np.round(np.mean(chunked_corrs), 3),
                                        np.round(np.std(chunked_corrs), 3),
                                        np.round(ttest_1samp(chunked_corrs, 0, nan_policy='omit').statistic, 3),
                                        np.round(ttest_1samp(chunked_corrs, 0, nan_policy='omit').pvalue, 3))))
                if self.movie.data_length - slot_end > step:
                    slot_start = slot_end  # index number to start
                    slot_end = slot_start + step  # index number to end
                else:
                    done = True
            all_ISC_data[s] = pd.DataFrame(subject_ISC_data,
                                           columns=['average_correlation', 'std_correlation', 't_statistic', 'p_value'])
        return all_ISC_data

    def subject_stats(self, corrs, func, incnan=False):
        # corrs = subjects_data_corrs
        # func = np.nanmean
        if self.is_external_group:
            subjects = ISC.external_group
        else:
            subjects = self.subjects
        keys = pd.Index(subjects)
        subjects1 = corrs.index.get_level_values('subject1').unique()
        subjects2 = corrs.index.get_level_values('subject2').unique()
        inboth = subjects1.intersection(subjects2).intersection(keys)
        onlys1 = subjects1.difference(subjects2).intersection(keys)
        onlys2 = subjects2.difference(subjects1).intersection(keys)

        both = [corrs.xs(sid, level='subject1').append(corrs.xs(sid, level='subject2')) for sid in inboth]
        head = [corrs.xs(sid, level='subject1') for sid in onlys1]
        tail = [corrs.xs(sid, level='subject2') for sid in onlys2]

        pool = both + head + tail
        means = [func(p) for p in pool if incnan or not np.all(np.isnan(p))]
        return pd.Series(means, index=keys)

    def plot_mean_correlation_subjects(self):
        subjects_data_corrs = self.subjects_data_corrs()
        bars = self.subject_stats(subjects_data_corrs, np.nanmean)
        errs = self.subject_stats(subjects_data_corrs, np.nanstd)
        # plot
        if self.is_external_group:
            subjects = ISC.external_group
        else:
            subjects = self.subjects
        xlab = 'Participant ID'
        ylab = "mean {} correlation".format(self.feature)
        xlabels = [str(s) for s in self.subjects]
        #title = "{}: mean {} correlation thoughts {}".format(self.facialExp, self.feature, self.movie.id)
        pltx = np.arange(subjects.size)
        plt.figure(figsize=(ISC.plotwidth, ISC.plotheight))
        plt.bar(pltx, bars, yerr=errs, capsize=3, color='darkblue')
        plt.xticks(pltx, xlabels, rotation=90)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.grid()
        title = "Mean correlation by participant of feature {} and movie {}".format(self.feature, self.movie.id)
        plt.title(title)
        plt.savefig(title + ".png")
        plt.show()

    def get_data_variables(self, chunk_size=1, alpha=0.05, is_dependent=True):
        # p = ISC(movie="m1", feature="Yaw", facialExp="head")
        data = pd.DataFrame([],
                            columns=["Participant", "Movie", "Hazards", "Mean_correlation", "Mean_chunked_correlation",
                                     "Sig_mean_correlation", "Turnings", "STD_{}".format(self.feature),
                                     "Road_experience"])
        subjects_data_corrs = self.subjects_data_corrs()
        all_ISC_data = self.subjects_data_corrs_by_chunk_size(chunk_size)
        mean_corrs = self.subject_stats(subjects_data_corrs, np.nanmean)
        if self.is_external_group:
            subjects = ISC.external_group
        else:
            subjects = self.subjects
        for s in subjects:
            # s = subjects[0]
            print(s)
            mean_corr = np.round(mean_corrs.loc[s], 2)
            s_ISC_data = all_ISC_data[s]
            mean_cunked_corrs = np.round(np.mean(s_ISC_data["average_correlation"]), 2)
            num_sig_pvalues = s_ISC_data[s_ISC_data["p_value"] < alpha].shape[0]
            num_hazard_identify, road_experience = self.movie.Events(
                self.movie.id).get_num_hazards_road_experience_by_participant(s, is_dependent=is_dependent)
            std_feature = np.round(np.std(self.get_processed_data_by_subjid(s)[self.feature]), 2)
            new_row = pd.DataFrame([(s, self.movie.id, num_hazard_identify, mean_corr, mean_cunked_corrs,
                                     num_sig_pvalues, ISC.movies_turnings_data[self.movie.id][0], std_feature,
                                     road_experience)], columns=data.columns)
            data = data.append(new_row)
        data["Road_experience"] = data["Road_experience"].apply(pd.to_numeric)
        data["Hazards"] = data["Hazards"].apply(pd.to_numeric)
        data["Sig_mean_correlation"] = data["Sig_mean_correlation"].apply(pd.to_numeric)
        data["Turnings"] = data["Turnings"].apply(pd.to_numeric)
        return data

    def get_corrs_variables(self, corrs_data, title=None):
        # corrs_data = f_data
        # plt.figure(figsize=(5, 5))
        # plt.subplots_adjust(top=0.88)
        plt.figure(figsize=(5, 5))
        sns.heatmap(corrs_data, annot=True, vmax=1, vmin=-1)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        if title is None:
            plt.suptitle("Correlation Matrix of {} and {}".format(self.movie.id, self.feature))
            plt.savefig("Correlation_Matrix_of_{}_{}_.png".format(self.movie.id, self.feature))
        else:
            plt.suptitle(title)
            plt.savefig(title + ".png")

    def plot_feature_distribution(self, movies_list=None):
        subjects_list = self.subjects
        for s in subjects_list:
            s_data = self.get_processed_data_by_subjid(s)
            if s == subjects_list[0]:
                all_data = s_data.copy()
                continue
            all_data = all_data.append(s_data)
        # Density Plot and Histogram of all arrival delays
        features = ISC.features_dict[self.facialExp]
        for f in features:
            plt.figure()
            all_data[f].hist()
            plt.suptitle("Histogram of {} and movie {}".format(f, self.movie.id))
            plt.savefig("Histogram of {} and movie {}".format(f, self.movie.id))
        if movies_list is not None:
            for m in movies_list:
                for s in subjects_list:
                    s_data = self.get_processed_data_by_subjid(s)
                    if s == subjects_list[0] and m == movies_list[0]:
                        all_data = s_data.copy()
                        continue
                    all_data = all_data.append(s_data)
            for f in features:
                plt.figure()
                all_data[f].hist()
                plt.suptitle("Histogram of {}".format(f))
                plt.savefig("Histogram of {}".format(f))

    def get_hazards_no_overlap(self):
        events_data = self.movie.events.event_times().reset_index(drop=True)
        cols_name = events_data.columns
        all_hazards_couples = list(combinations(cols_name, 2))
        overlaps_hazards = []
        for couple in all_hazards_couples:
            # couple = all_hazards_couples[0]
            start_time_list = [events_data[couple[0]].iloc[0], events_data[couple[1]].iloc[0]]
            end_time_list = [events_data[couple[0]].iloc[1], events_data[couple[1]].iloc[1]]
            latest_start = np.max(start_time_list)
            earliest_end = np.min(end_time_list)
            delta = earliest_end - latest_start
            if delta > 0:
                overlaps_hazards.append(couple)
                break
        if len(overlaps_hazards) == 0:
            new_events_data = events_data.sort_values(by=events_data.index[0], axis=1)
            return events_data
        new_cols_name = list(set(cols_name) - set(overlaps_hazards[0]))
        new_events_data = events_data[new_cols_name]
        new_hazard = pd.DataFrame([np.min(start_time_list), np.max(end_time_list)], columns=["new_hazard"])
        new_events_data = pd.concat([new_events_data, new_hazard], axis=1)
        new_events_data = new_events_data.sort_values(by=new_events_data.index[0], axis=1)
        return new_events_data

    def get_time_section_hazards_and_no_hazards_data(self):
        #p = ISC(movie="m6")
        hazards_data = self.get_hazards_no_overlap()
        movie_legth_sec = self.movie.data_length / 30
        non_hazards_data = pd.DataFrame()
        for i, hazard in enumerate(hazards_data.columns):
            #hazard = hazards_data.columns[-1]
            start = hazards_data.loc[0, hazard]
            end = hazards_data.loc[1, hazard]
            if hazard == hazards_data.columns[-1]:
                if end >= movie_legth_sec:
                    end = np.min([end, movie_legth_sec])
                    hazards_data.loc[1, hazard] = end
                    continue
                else:
                    non_hazard_section = pd.DataFrame([end, movie_legth_sec])
                    non_hazards_data = pd.concat([non_hazards_data, non_hazard_section], axis=1)
                    continue
            next_hazard = hazards_data.columns[i + 1]
            next_hazard_start = hazards_data.loc[0, next_hazard]
            non_hazard_section = pd.DataFrame([end, next_hazard_start])
            non_hazards_data = pd.concat([non_hazards_data, non_hazard_section], axis=1)
            if hazard == hazards_data.columns[0]:
                if start == 0:
                    continue
                else:
                    non_hazard_section = pd.DataFrame([0, start])
                    non_hazards_data = pd.concat([non_hazards_data, non_hazard_section], axis=1)
                    continue
        non_hazards_data = non_hazards_data.sort_values(by=non_hazards_data.index[0], axis=1)
        non_hazards_data.index = ["start", "end"]
        hazards_data.index = ["start", "end"]
        return non_hazards_data, hazards_data

    def get_ISC_section_by_data(self, data):
        # data = non_hazards_data
        ISC_data = np.empty((0, 4))  # cols names: ['average_correlation', 'std_correlation', 't_statistic', 'p_value']
        fps = self.movie.fps
        if self.is_external_group:
            subjects_list = ISC.external_group
        else:
            subjects_list = self.subjects
        for col in range(data.shape[1]):
            # col = 1
            start_hazard = data.iloc[0, col]
            end_hazard = data.iloc[1, col]
            current_slot_start = int(start_hazard * fps)
            current_slot_end = int(end_hazard * fps)
            current_pearson_corrs_data = self.get_correlation_data(subjects=subjects_list,
                                                                slot_start=current_slot_start,
                                                                slot_end=current_slot_end)
            ISC_data = np.vstack(
                (ISC_data, (np.round(current_pearson_corrs_data.mean(), 3), np.round(current_pearson_corrs_data.std(), 3),
                            np.round(ttest_1samp(current_pearson_corrs_data, 0, nan_policy='omit').statistic, 3),
                            np.round(ttest_1samp(current_pearson_corrs_data, 0, nan_policy='omit').pvalue, 3))))

        return pd.DataFrame(ISC_data, columns=['average_correlation', 'std_correlation', 't_statistic', 'p_value'],
                            dtype='float64')

    def get_ISC_section_hazards_and_no_hazards_data(self):
        non_hazards_data, hazards_data = self.get_time_section_hazards_and_no_hazards_data()
        if non_hazards_data.shape[0] != 0:
            ISC_data_non_hazard = self.get_ISC_section_by_data(non_hazards_data)
        else:
            ISC_data_non_hazard = pd.DataFrame()
        if hazards_data.shape[0] != 0:
            ISC_data_hazard = self.get_ISC_section_by_data(hazards_data)
        else:
            ISC_data_hazard = pd.DataFrame()
        return ISC_data_non_hazard, ISC_data_hazard

    def get_all_section_by_start_end(self, data):
        # data = non_hazards_data
        if self.is_external_group:
            subjects_list = ISC.external_group
        else:
            subjects_list = self.subjects
        fps = self.movie.fps
        all_data = pd.DataFrame([], columns=["Participant", self.feature])
        for col in range(data.shape[1]):
            # col = 0
            start_hazard = data.iloc[0, col]
            end_hazard = data.iloc[1, col]
            current_slot_start = int(start_hazard * fps)
            current_slot_end = int(end_hazard * fps)
            for s in subjects_list:
                # s = subjects_list[3]
                s_data = self.get_processed_data_by_subjid(s)[self.feature].iloc[current_slot_start: current_slot_end]
                s_data = pd.DataFrame(s_data, columns=[self.feature])
                s_data["Participant"] = s
                all_data = all_data.append(s_data)
        return all_data

    def plot_ISC_section_hazards_and_no_hazards_data(self, is_mean_of_mean=True):
        if is_mean_of_mean:
            ISC_data_non_hazard, ISC_data_hazard = self.get_ISC_section_hazards_and_no_hazards_data()
            mean_hazards = np.mean(ISC_data_hazard["average_correlation"].to_numpy())
            mean_non_hazards = np.mean(ISC_data_hazard["average_correlation"].to_numpy())
            plt.figure()
            plt.bar(x="hazards", height=mean_hazards)
            plt.bar(x="non-hazards", height=mean_non_hazards)
            title = "Mean ISC for hazard and non-hazard sections of {} and {}".format(self.movie.id, self.feature)
            plt.suptitle(title)
            plt.ylabel("Mean ISC of {}".format(self.feature))
            plt.savefig(title + ".png")
        else:
            non_hazards_data, hazards_data = self.get_time_section_hazards_and_no_hazards_data()
            all_hazards_frames_by_subject = self.get_all_section_by_start_end(hazards_data)
            all_non_hazards_frames_by_subject = self.get_all_section_by_start_end(non_hazards_data)
            subjects_list = np.unique(all_non_hazards_frames_by_subject["Participant"])
            hazards_corr_data = self.get_all_corrs_couples(subjects_list, all_hazards_frames_by_subject)
            non_hazards_corr_data = self.get_all_corrs_couples(subjects_list, all_non_hazards_frames_by_subject)
            mean_hazards = np.mean(hazards_corr_data)
            mean_non_hazards = np.mean(non_hazards_corr_data)
            plt.figure()
            plt.bar(x="hazards", height=mean_hazards)
            plt.bar(x="non-hazards", height=mean_non_hazards)
            title = "Mean ISC for hazard and non-hazard of {} and {} - without sections".format(self.movie.id, self.feature)
            plt.suptitle(title)
            plt.ylabel("Mean ISC")
            plt.savefig(title + ".png")

    def get_all_corrs_couples(self, subjects_list, data):
        # data = all_data_non_hazard
        all_data_couples = list(combinations(subjects_list, 2))
        corr_data = pd.Series(index=pd.MultiIndex.from_tuples(all_data_couples, names=['subject1', 'subject2']), dtype='float64')
        for subject1, subject2 in corr_data.index:
            #subject1, subject2 = corr_data.index[0]
            subject1_data = data[data["Participant"] == subject1][self.feature]
            subject2_data = data[data["Participant"] == subject2][self.feature]
            corr = np.corrcoef(subject1_data, subject2_data)[0, 1]
            corr_data[subject1, subject2] = corr
        return corr_data

    def plot_ISC_section_hazards_and_no_hazards_data_all_movies(self, movies_list, is_mean_of_mean=True):
        if not is_mean_of_mean:
            all_data_hazard = pd.DataFrame(columns=["Participant", self.feature])
            all_data_non_hazard = pd.DataFrame(columns=["Participant", self.feature])
            for m in movies_list:
                # m = movies_list[0]
                ISC_instance = ISC(movie=m, feature=self.feature)
                non_hazards_data, hazards_data = ISC_instance.get_time_section_hazards_and_no_hazards_data()
                all_hazards_frames_by_subject = self.get_all_section_by_start_end(hazards_data)
                all_non_hazards_frames_by_subject = self.get_all_section_by_start_end(non_hazards_data)
                if hazards_data.shape[0] != 0:
                    all_data_hazard = all_data_hazard.append(all_hazards_frames_by_subject)
                if non_hazards_data.shape[0] != 0:
                    all_data_non_hazard = all_data_non_hazard.append(all_non_hazards_frames_by_subject)
            subjects_list = np.unique(all_data_non_hazard["Participant"])
            hazards_corr_data = self.get_all_corrs_couples(subjects_list, all_data_hazard)
            non_hazards_corr_data = self.get_all_corrs_couples(subjects_list, all_data_non_hazard)
            mean_hazards = np.mean(hazards_corr_data)
            mean_non_hazards = np.mean(non_hazards_corr_data)
            plt.figure()
            plt.bar(x="hazards", height=mean_hazards)
            plt.bar(x="non-hazards", height=mean_non_hazards)
            title = "Mean ISC for hazard and non-hazard of {} - without sections".format(self.feature)
            plt.suptitle(title)
            plt.ylabel("Mean ISC")
            plt.savefig(title + ".png")
        else:
            ISC_all_data_hazard = pd.DataFrame()
            ISC_all_data_non_hazard = pd.DataFrame()
            for m in movies_list:
                #m = movies_list[0]
                #m="m7"
                ISC_instance = ISC(movie=m, feature=self.feature)
                ISC_data_non_hazard, ISC_data_hazard = ISC_instance.get_ISC_section_hazards_and_no_hazards_data()
                if ISC_data_hazard.shape[0] != 0:
                    ISC_all_data_hazard = ISC_all_data_hazard.append(pd.DataFrame(ISC_data_hazard["average_correlation"], dtype="float"))
                if ISC_data_non_hazard.shape[0] != 0:
                    ISC_all_data_non_hazard = ISC_all_data_non_hazard.append(pd.DataFrame(ISC_data_non_hazard["average_correlation"], dtype="float"))
            mean_hazards = np.mean(ISC_all_data_hazard.to_numpy())
            mean_non_hazards = np.mean(ISC_all_data_non_hazard.to_numpy())
            plt.figure()
            plt.bar(x="hazards", height=mean_hazards)
            plt.bar(x="non-hazards", height=mean_non_hazards)
            title = "Mean ISC for hazard and non-hazard sections of {}".format(self.feature)
            plt.suptitle(title)
            plt.ylabel("Mean ISC")
            plt.savefig(title + ".png")

    def get_data_natural_expression(self, threshold=50):
        # p = ISC(movie="m1", feature="Joy", facialExp="facial")
        subjects_list = self.subjects
        data = pd.DataFrame([], columns=["Movie", "Participant", "total frames", "total frames with natural", "total frames with natural emotions", "total frames with natural AUs"])
        for s in subjects_list:
            # s = subjects_list[0]
            print(s)
            s_data = self.get_processed_data_by_subjid(s)[ISC.facial]
            total_frames = s_data.shape[0]
            Participant = s
            Movie = self.movie.id
            total_frames_with_natural = s_data[(s_data > threshold).all(axis=1)].shape[0]
            if total_frames_with_natural > 0:
                total_frames_with_natural_EM = s_data[(s_data > threshold).all(axis=1)][ISC.EM].shape[0]
                total_frames_with_natural_AUs = s_data[(s_data > threshold).all(axis=1)][ISC.AUs].shape[0]
            else:
                total_frames_with_natural_EM = 0
                total_frames_with_natural_AUs = 0
            new_row = pd.DataFrame([(Movie, Participant, total_frames, total_frames_with_natural, total_frames_with_natural_EM, total_frames_with_natural_AUs)], columns=["Movie", "Participant", "total frames", "total frames with natural", "total frames with natural emotions", "total frames with natural AUs"])
            data = data.append(new_row)
        return data

    def get_data_natural_expression_by_feature(self, threshold=50):
        # p = ISC(movie="m1", feature="Joy", facialExp="facial")
        subjects_list = self.subjects
        data = pd.DataFrame([], columns=["Movie", "Participant", "feature", "total frames", "total frames with natural"])
        for s in subjects_list:
            # s = subjects_list[0]
            print(s)
            s_data = self.get_processed_data_by_subjid(s)[ISC.facial]
            total_frames = s_data.shape[0]
            Participant = s
            Movie = self.movie.id
            for f in ISC.facial:
                f_s_data = s_data[f]
                total_frames_with_natural = f_s_data[f_s_data > threshold].shape[0]
                feature = f
                new_row = pd.DataFrame([(Movie, Participant, feature, total_frames, total_frames_with_natural)], columns=["Movie", "Participant", "feature", "total frames", "total frames with natural"])
                data = data.append(new_row)
        return data

    def plot_emotions_by_participant_movie(self, participant):
        #participant = '237'
        #p=ISC(movie="m15", feature="Joy", facialExp="facial")
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(10, 5))
        data = self.get_processed_data_by_subjid(participant)
        data_2_plot = data[ISC.EM]
        media_time_movie = self.reset_media_time_col(data["MediaTime"])
        _, media_time_movie_sec = divmod(media_time_movie.values / 1000, 60)
        length_movie_sec = math.ceil(self.movie.data_length / self.movie.fps)
        #x_values = list(range(0, data_2_plot.shape[0]))
        sns.set_style('whitegrid')
        for feature in ISC.EM:
            plt.plot(media_time_movie_sec, data_2_plot[feature].to_numpy(), label=feature)
        plt.xlabel("Seconds")
        plt.ylabel("Emotions")
        plt.xticks(range(0, length_movie_sec, 2))
        plt.legend(prop=dict(size=10), loc=(0.98, 0.5))
        title = "Emotions During Movie {} of Participant {}".format(self.movie.id, participant)
        plt.title(title)
        plt.savefig(title + ".png")

    def plot_feature_by_participant_movie_feature(self, participant):
        #participant = '237'
        #p=ISC(movie="m15", feature="Joy", facialExp="facial")
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(8, 5))
        data = self.get_processed_data_by_subjid(participant)
        media_time_movie = self.reset_media_time_col(data["MediaTime"])
        _, media_time_movie_sec = divmod(media_time_movie.values / 1000, 60)
        length_movie_sec = math.ceil(self.movie.data_length / self.movie.fps)
        #x_values = list(range(0, data_2_plot.shape[0]))
        sns.set_style('whitegrid')
        plt.plot(media_time_movie_sec, data[self.feature].to_numpy())
        plt.xlabel("Seconds")
        plt.ylabel(self.feature)
        plt.xticks(range(0, length_movie_sec, 2))
        title = "{} During Movie {} of Participant {}".format(self.feature, self.movie.id, participant)
        plt.title(title)
        plt.savefig(title + ".png")


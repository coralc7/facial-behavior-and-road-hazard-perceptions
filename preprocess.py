import glob
import h5py
import io
import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from create_rawData import RawDataCreation


class PreProc_missingValues:
    datadir = RawDataCreation.datadir  # gets the directory of "data" folder
    rawDatadir = RawDataCreation.rawDatadir
    data2PreProcess_dir = os.path.join(rawDatadir, 'data2PreProcess')  # gets the directory of "data2PreProcess" folder in "raw_data" folderdemographicData
    AU = ['Brow Furrow', 'Brow Raise', 'Lip Corner Depressor', 'InnerBrowRaise',
          'EyeClosure', 'NoseWrinkle', 'UpperLipRaise', 'LipSuck', 'LipPress', 'MouthOpen', 'ChinRaise',
          'Smirk', 'LipPucker', 'Cheek Raise', 'Dimpler', 'Eye Widen', 'Lid Tighten', 'Lip Stretch',
          'Jaw Drop']  # Action Units
    EM = ['Anger', 'Sadness', 'Disgust', 'Joy', 'Surprise', 'Fear', 'Contempt']  # emotions
    HM = ['Pitch', 'Yaw', 'Roll']  # head movements
    # generalCols = ['Name', 'FrameIndex', 'LiveMarker', 'MediaTime']
    limits = {
        "head": [-100, 100],
        # The upper and lower limit of the normal values of Yaw, Pitch, Roll (This is the same for each)
        "AUs": [0, 100],  # The upper and lower limit of the normal values
        "emotions": [0, 100],
        # The upper and lower limit of the normal values emotions like Disgust, Joy etc (This is the same for each)
        "facial": [0, 100]  # The upper and lower limit of the normal values emotions and AUs
    }
    chosenCols = {"head": HM, "AUs": AU, "emotions": EM, "facial": EM + AU}
    fpat = RawDataCreation.fpat

    def __init__(self, min_valid=5, max_hole_size=10, facialExp="head"):
        self.subjIDs = sorted(list(set([re.findall(PreProc_missingValues.fpat, f)[0] for f in glob.glob(
            os.path.join(PreProc_missingValues.data2PreProcess_dir, "m*", "*.txt"))])))  # gets all the subjects' ID
        self.mIDs = sorted(list(set([mID for mID in os.listdir(PreProc_missingValues.data2PreProcess_dir) if
                                     (mID.startswith("m")) & (~mID.endswith(".csv"))])))  # gets all the movies' ID
        self.min_valid = min_valid  # 1 condition to do interpolation - valid values before "hole" (hole = missing values in a raw, I call it "section")
        self.max_hole_size = max_hole_size  # 2 condition to do interpolation - max hole length
        self.facialExp = facialExp  # The facial expression we want to analyze

    def getData_NaNData(self, subjectID, mIDdir, alsoNaNData=False):
        sraw_dir = os.path.join(mIDdir, subjectID + ".txt")
        relevantCols = ['Name', 'FrameIndex'] + PreProc.chosenCols[self.facialExp]
        s_data = pd.read_csv(sraw_dir, sep="\t", usecols=relevantCols)
        if alsoNaNData:
            if self.facialExp == "head":
                s_data_NaN = s_data[s_data[PreProc.chosenCols[self.facialExp]].isnull().all(axis=1)]
            else:
                cond_null = s_data[PreProc.chosenCols[self.facialExp]].isnull()
                cond_zero = s_data[PreProc.chosenCols[self.facialExp]] == 0
                s_data_NaN = s_data[(cond_null | cond_zero).all(axis=1)]
            return s_data, s_data_NaN
        else:
            return s_data

    def missingValuesMap(self, Destinationfolder="data2PreProcess"):
        missingValuesMap = pd.DataFrame(
            columns=["Movie", "ID", "SectionID", "startIndx", "endIndx", "TotalMissingFrames"])
        for movie in self.mIDs:
            # movie = "m3"
            # print(movie)
            if Destinationfolder == "data2PreProcess":
                mIDdir = os.path.join(PreProc_missingValues.data2PreProcess_dir, movie)
            elif Destinationfolder == "rawDatadir":
                mIDdir = os.path.join(PreProc_missingValues.rawDatadir, movie)
            else:
                mIDdir = os.path.join(PreProc_missingValues.datadir, movie)
            fpat = PreProc_missingValues.fpat
            subjids = sorted(list(set([re.findall(fpat, f)[0] for f in glob.glob(os.path.join(mIDdir, "*.txt"))])))
            for s in subjids:
                # s="215"
                # print(s)
                s_data, s_data_NaN = self.getData_NaNData(s, mIDdir, alsoNaNData=True)
                if s_data_NaN.empty:
                    continue
                s_dataDiff = s_data_NaN.loc[:, "FrameIndex"].diff()
                if s_dataDiff[s_dataDiff != 1].size == 1:  # checking if all NaN is 1 section in a row
                    TotalMissingValues = int(s_data_NaN.shape[0])
                    s_data_NaN.reset_index(inplace=True)
                    newRow = pd.DataFrame(data=[(movie, s, 1, int(s_data_NaN.loc[:, "FrameIndex"].iloc[0]),
                                                 int(s_data_NaN.loc[:, "FrameIndex"].iloc[-1]), TotalMissingValues)],
                                          columns=["Movie", "ID", "SectionID", "startIndx", "endIndx",
                                                   "TotalMissingFrames"])
                    missingValuesMap = missingValuesMap.append(newRow)
                else:  # there is more than 1 section of NaN values
                    combine = {'Diff': s_dataDiff, 'FrameIndex': s_data.loc[:, "FrameIndex"]}
                    s_data_Diff_FrameIndex = pd.DataFrame(combine)

                    s_data_Diff_FrameIndex.dropna(inplace=True)
                    s_data_Diff_FrameIndex.reset_index(inplace=True)

                    startListby_frameIndex = [s_data_NaN.iloc[0, 1]] + s_data_Diff_FrameIndex.iloc[
                        s_data_Diff_FrameIndex.index[(s_data_Diff_FrameIndex[
                                                          'Diff'] != 1)].tolist(), 2].tolist()  # the index of the start of each section of NaN in a row
                    indexEndSectionNaN = s_data_Diff_FrameIndex.index[(s_data_Diff_FrameIndex['Diff'] != 1)] - 1
                    if indexEndSectionNaN[0] == -1:  # true = the size of first section is 1
                        indexEndSectionNaN = indexEndSectionNaN[1:]
                        endListby_frameIndex = [s_data_NaN.iloc[0, 1]] + s_data_Diff_FrameIndex.iloc[
                            indexEndSectionNaN.tolist(), 2].tolist() + [s_data_NaN.iloc[-1, 1]]
                    else:
                        endListby_frameIndex = s_data_Diff_FrameIndex.iloc[indexEndSectionNaN.tolist(), 2].tolist() + [
                            s_data_NaN.iloc[-1, 1]]

                    combine_StartEnd = {'start': startListby_frameIndex, 'end': endListby_frameIndex}
                    df_startEnd_sections = pd.DataFrame(combine_StartEnd)
                    df_startEnd_sections["TotalMissingFrames"] = df_startEnd_sections['end'] - df_startEnd_sections[
                        'start'] + 1
                    newRows = pd.DataFrame()
                    newRows["TotalMissingFrames"] = df_startEnd_sections["TotalMissingFrames"].astype("int64")
                    newRows["SectionID"] = range(1, df_startEnd_sections.shape[0] + 1)
                    newRows["startIndx"] = df_startEnd_sections['start']
                    newRows["endIndx"] = df_startEnd_sections['end']
                    newRows["Movie"] = movie
                    newRows["ID"] = int(s)
                    missingValuesMap = missingValuesMap.append(newRows, sort=False)
        # save missingValuesMap
        if Destinationfolder == "datadir":
            writeFile = os.path.join(PreProc_missingValues.datadir, "missingValuesMap_Final_" + self.facialExp + ".csv")
        elif Destinationfolder == "rawDatadir":
            writeFile = os.path.join(PreProc_missingValues.datadir,
                                     "missingValuesMap_notexcatBegin_" + self.facialExp + ".csv")
        else:
            writeFile = os.path.join(PreProc_missingValues.datadir, "missingValuesMap_" + self.facialExp + ".csv")
        missingValuesMap.to_csv(writeFile, index=False)
        return missingValuesMap

    def elementOccurrences(self, Occurrences, element):
        """
        returns a list of (start,length) tuples representing
        the occurrences of the element in Occurrences. for example:
        Occurrences = [1,1,1,2,2,1,1]
        returns [(0,3),(5,2)] for element = 1
        and [(4,2)] for element = 2
        """
        pairs = list()
        inds = np.where(Occurrences == element)[0]  # end
        if len(inds) == 0:  # There is no instance of the element
            return pairs
        breaks = np.where(np.diff(inds) > 1)[0]
        if len(breaks) == 0:  # there is only 1 hole
            pairs.append((inds[0], len(inds)))
            return pairs
        start = 0
        for b in breaks:
            length = b - start + 1
            pairs.append((inds[start], length))
            start += length
        if start < len(inds):
            pairs.append((inds[start], inds[-1] - inds[start] + 1))
        return pairs

    def get_max_hole_and_precentage_rows_any_NaN(self, data):
        data = data.values
        if self.facialExp != "head":
            data[(data == 0).all(axis=1)] = np.nan
        rows_labels_is_any_NaN = np.isnan(data).all(axis=1).astype(np.int)
        holes = self.elementOccurrences(Occurrences=rows_labels_is_any_NaN, element=1)
        if len(holes) == 0:
            max_hole = 0
        else:
            indexs_list, hole_sizes_list = zip(*holes)
            max_hole = np.max(hole_sizes_list)
        percentage_rows_any_NaN = np.sum(rows_labels_is_any_NaN) / data.shape[0]
        percentage_max_hole = max_hole / data.shape[0]
        return max_hole, np.round(percentage_max_hole, 2), np.round(percentage_rows_any_NaN, 2)

    def manipulate(self, data, isValidAfterHole=True, method='mix', limit_direction='both', order=3, polyWeight=0.5):
        colnames = data.columns
        data = data.values  # from pd to np
        if self.facialExp != "head":
            data[(data == 0).all(axis=1)] = np.nan
        holes = self.elementOccurrences(Occurrences=np.isnan(data).all(axis=1).astype(np.int), element=1)
        lastnan = -1
        isLastHoleNaN = False
        for i, (start, length) in enumerate(holes):
            # i=28
            # start = holes[i][0]
            # length = holes[i][1]
            end = start + length - 1
            # hole is too big
            if length > self.max_hole_size:
                lastnan = end
                isLastHoleNaN = True
                continue

            if i == len(holes) - 1:
                if isValidAfterHole:
                    if holes[i][0] - data.shape[0] < self.min_valid:
                        lastnan = end
                        isLastHoleNaN = True
                        continue
            else:
                if isValidAfterHole:
                    if holes[i + 1][0] - end - 1 < self.min_valid:
                        lastnan = end
                        isLastHoleNaN = True
                        continue
            if i == 0:
                if start - 1 < self.min_valid:  # preceding stretch of good values is too short for the first hole. In case of i==0 -> start-1 = data[0:start].shape[0]
                    lastnan = end
                    isLastHoleNaN = True
                    continue
            else:
                if isLastHoleNaN:
                    firstValidFrame_sequence = sum(holes[i - 1])  # last frame of the last hole is NaN
                else:
                    # prevend = sum(holes[i - 1]) - 1  #last frame of the last hole is not NaN
                    firstValidFrame_sequence = lastnan + 1  # last frame of the last hole is not NaN (or all the holes till now are fixed)
                # preceding stretch of good values is too short
                if start - firstValidFrame_sequence < self.min_valid:
                    lastnan = end
                    isLastHoleNaN = True
                    continue
            start2interpolate = lastnan + 1
            end2interpolate = holes[i + 1][0] if i < len(holes) - 1 else data.shape[
                0]  # I want all the valid data after this hole (= until the next hole), if this the last hole so I want all the rest of the data (This is certainly a valid data_
            isLastHoleNaN = False
            if method == "mix":
                polyData = pd.DataFrame(data[start2interpolate:end2interpolate]).interpolate(method="polynomial",
                                                                                             limit_direction=limit_direction,
                                                                                             order=order).values
                linearData = pd.DataFrame(data[start2interpolate:end2interpolate]).interpolate(method="linear",
                                                                                               limit_direction=limit_direction,
                                                                                               order=None).values
                data[start2interpolate:end2interpolate] = polyWeight * polyData + (1 - polyWeight) * linearData
            else:
                data[start2interpolate:end2interpolate] = pd.DataFrame(
                    data[start2interpolate:end2interpolate]).interpolate(method=method, limit_direction=limit_direction,
                                                                         order=order).values
        # short segments of NaN are replaced with 0, so that k-means can run
        # the segmentation object will later gloss over them when producing the word sequence
        # feat[np.isnan(feat)] = 0
        # bit ugly.. but works
        data = pd.DataFrame(data.clip(PreProc.limits[self.facialExp][0], PreProc.limits[self.facialExp][1]),
                            columns=colnames)  # SELF Assigns values outside boundary to boundary values
        if self.facialExp != "head":
            data.loc[data.sum(axis=1) == 0] = np.nan  # each row (sum of all cols) who equals to zero assign to NaN
        return data, holes

    def NaN_raport(self, Destinationfolder="datadir"):
        if Destinationfolder == "rawDatadir":
            Destinationfolder_dir = RawDataCreation.rawDatadir
        else:
            Destinationfolder_dir = RawDataCreation.datadir
        NaN_report = pd.DataFrame(columns=["Movie", "ID", "FacialExspression"])
        mIDs = sorted(list(
            set([mID for mID in os.listdir(Destinationfolder_dir) if (mID.startswith("m")) & (~mID.endswith(".csv"))])))
        # mIDs = [mfile2mid(f) for f in listdir]
        # MV =PreProc_missingValues(facialExp="head")
        for movie in mIDs:
            # movie="m3"
            # print("####_" + movie + "_####")
            mIDdir = os.path.join(Destinationfolder_dir, movie)
            fpat = PreProc_missingValues.fpat
            subjids = sorted(list(set([re.findall(fpat, f)[0] for f in glob.glob(os.path.join(mIDdir, "*.txt"))])))
            for s in subjids:
                # s="215"
                # print(s)
                # sraw_dir = os.path.join(mIDdir, s + ".txt")
                s_data, s_data_NaN = self.getData_NaNData(s, mIDdir, alsoNaNData=True)
                data = s_data[PreProc_missingValues.chosenCols[self.facialExp]]
                if s_data_NaN.shape[0] > 0:
                    newRow = pd.DataFrame(data=[(movie, s, self.facialExp)],
                                          columns=["Movie", "ID", "FacialExspression"])
                    NaN_report = NaN_report.append(newRow)
        writeFile = os.path.join(PreProc_missingValues.datadir, "NaN_Report_excatSplit" + self.facialExp + ".csv")
        NaN_report.to_csv(writeFile, index=False)
        return NaN_report

    def plotGraphComparison(self, FrameIndex, originData, ProcessedData, features, movieID, participantID, title=None):
        if self.facialExp != "head":
            originData[originData == 0] = np.nan
            ProcessedData[ProcessedData == 0] = np.nan
        for f in features:
            if f == "FrameIndex":
                continue
            # f='Pitch'
            plt.figure()
            plt.plot(FrameIndex.values, ProcessedData[f].values, marker='', color='blue', linestyle='-',
                     label="Processed Data")
            plt.plot(FrameIndex.values, originData[f].values, marker='', color='red', linestyle='-',
                     label="Original Data")
            plt.legend()
            plt.ylabel("{:s}".format(f))
            plt.xlabel("Frame Index")
            plt.title("'{:s}': Movie '{:s}', Participant '{:s}''".format(title, movieID, participantID))
            plt.show()

    def graphComparison(self, movieID, participantID, title=None):
        if (movieID not in self.mIDs) | (participantID not in self.subjIDs):
            print("The movie ID '{:s}' or participant ID '{:s}' does not exist. Please try new ones.".format(movieID,
                                                                                                             participantID))
            # return
        # movieID = "m3"
        # participantID = "201"
        originData_dir = os.path.join(PreProc_missingValues.data2PreProcess_dir, movieID, participantID + ".txt")
        ProcessedData_dir = os.path.join(PreProc_missingValues.rawDatadir, movieID, participantID + ".txt")
        relevantCols = ['FrameIndex'] + PreProc.chosenCols[self.facialExp]
        ProcessedData = pd.read_csv(ProcessedData_dir, sep="\t", usecols=relevantCols)
        originData = pd.read_csv(originData_dir, sep="\t", usecols=relevantCols)
        FrameIndex = originData["FrameIndex"].copy()
        self.plotGraphComparison(FrameIndex, originData, ProcessedData, relevantCols, movieID, participantID,
                                 title=title)

    def missingValues_process(self, movie, subjID, isValidAfterHole=True, method='mix', limit_direction='both', order=3,
                              polyWeight=0.5, isSaving=False, isPlot=False):
        # movie = "m1"
        mid_dir_rawData = os.path.join(PreProc_missingValues.rawDatadir, movie)
        if os.path.isdir(mid_dir_rawData):
            mIDdir = os.path.join(PreProc_missingValues.rawDatadir, movie)
        else:
            mIDdir = os.path.join(PreProc_missingValues.data2PreProcess_dir, movie)
        if not os.path.isdir(mIDdir):
            print("can't find raw data dir for {:s} at {:s}. skipping.\n".format(movie, mIDdir))
            return
        # subjID="201"
        subj_dir = os.path.join(mIDdir, subjID + ".txt")
        if not os.path.isfile(subj_dir):
            print("\tcan't find raw data file for subject {:s}. at {:s}. skipping.\n".format(subjID, subj_dir))
            return
        if os.path.isfile(os.path.join(PreProc_missingValues.rawDatadir, movie, subjID + ".txt")):
            subj_dir = os.path.join(PreProc_missingValues.rawDatadir, movie, subjID + ".txt")
        else:
            subj_dir = os.path.join(PreProc_missingValues.data2PreProcess_dir, movie, subjID + ".txt")
        s_data = pd.read_csv(subj_dir, sep="\t")
        features = PreProc_missingValues.chosenCols[self.facialExp]
        OriginData = s_data[features].copy()
        if self.facialExp != "head":
            OriginData[(OriginData == 0).all(axis=1)] = np.nan
            tempMissingValues_instance = PreProc_missingValues(min_valid=2, max_hole_size=1, facialExp=self.facialExp)
            ProcessedData_firstTime, _ = tempMissingValues_instance.manipulate(OriginData.copy(),
                                                                               isValidAfterHole=isValidAfterHole,
                                                                               method=method,
                                                                               limit_direction=limit_direction,
                                                                               order=order, polyWeight=polyWeight)
            ProcessedData, _ = self.manipulate(ProcessedData_firstTime.copy(), isValidAfterHole=isValidAfterHole,
                                               method=method, limit_direction=limit_direction, order=order,
                                               polyWeight=polyWeight)
        else:
            ProcessedData, _ = self.manipulate(OriginData.copy(), isValidAfterHole=isValidAfterHole, method=method,
                                               limit_direction=limit_direction, order=order, polyWeight=polyWeight)
        if isSaving:
            createData = RawDataCreation()  # instance of another calss
            createData.saveDataByMovies(data=s_data, movie=movie, participant=subjID, Destinationfolder="rawDatadir")
        if isPlot:
            self.plotGraphComparison(s_data["FrameIndex"], s_data, ProcessedData, features, movie, subjID,
                                     title="Missing Values")
        return s_data, ProcessedData


class PreProc:
    datadir = RawDataCreation.datadir  # gets the directory of "data" folder
    rawDatadir = RawDataCreation.rawDatadir  # gets the directory of "raw_data" folder in "data" folder
    data2PreProcess_dir = os.path.join(rawDatadir, 'data2PreProcess')  # gets the directory of "data2PreProcess" folder in "raw_data" folder
    AU = PreProc_missingValues.AU  # Action Units
    EM = PreProc_missingValues.EM  # emotions
    HM = PreProc_missingValues.HM  # head movements
    generalCols = ['Name', 'FrameIndex', 'LiveMarker', 'MediaTime']
    chosenCols = PreProc_missingValues.chosenCols
    limits = PreProc_missingValues.limits
    fpat = PreProc_missingValues.fpat
    generalCols = ['Name', 'FrameIndex', 'LiveMarker', 'MediaTime']
    fps = 30

    def __init__(self, smoothing_window=4, maxValid=5, facialExp="head"):
        self.mids = sorted(list(set([mID for mID in os.listdir(PreProc.data2PreProcess_dir) if
                                     (mID.startswith("m")) & (~mID.endswith(".csv"))])))  # gets all the movies' ID
        self.subjIDs = sorted(list(set([re.findall(PreProc.fpat, f)[0] for f in glob.glob(
            os.path.join(PreProc.data2PreProcess_dir, "m*", "*.txt"))])))  # gets all the subjects' ID
        self.smoothing_window = smoothing_window
        self.smoothing_maxValid = maxValid
        self.facialExp = facialExp  # The facial expression we want to analyze

    def fillNaN_mean(self, data, holes):
        data = data.values
        for i, (start, length) in enumerate(holes):
            # i=0
            # start = holes[i][0]
            # length = holes[i][1]
            end = start + length - 1
            if i == 0:
                if start - 0 < self.smoothing_maxValid:  # לבדוק כאשר אין NAN בהתחלה
                    before = data[0: start]
                else:  #
                    before = data[start - self.smoothing_maxValid: start]
            else:
                lastValid = holes[i - 1][0] + holes[i - 1][1]
                if start - lastValid < self.smoothing_maxValid:  # valid data before hole
                    before = data[lastValid:start]
                else:
                    before = data[start - self.smoothing_maxValid:start]

            if i == len(holes) - 1:  # last element in holes list
                if data.shape[0] - (end + 1) < self.smoothing_maxValid:  # valid data after hole
                    after = data[end + 1: data.shape[0]]
                else:
                    after = data[end + 1: end + 1 + self.smoothing_maxValid]
            else:
                if holes[i + 1][0] - (end + 1) < self.smoothing_maxValid:  # valid data after hole
                    after = data[end + 1: holes[i + 1][0]]
                else:
                    after = data[end + 1: end + 1 + self.smoothing_maxValid]
            if (len(before) == 0) & (len(after) != 0):
                meanAfter = np.mean(after)
                data[start: end + 1] = meanAfter
            elif (len(before) != 0) & (len(after) == 0):
                meanBefore = np.mean(before)
                data[start: end + 1] = meanBefore
            else:
                meanBefore = np.mean(before)
                meanAfter = np.mean(after)
                mean = np.mean([meanBefore, meanAfter])
                data[start: end + 1] = mean
        return pd.DataFrame(data)

    def smoothing(self, data, holes):
        # data = data.loc[:, col].copy()
        data = self.fillNaN_mean(data, holes)
        data = pd.DataFrame(data)
        data_processed = data.rolling(self.smoothing_window, min_periods=1).mean()  # moving average
        newHoles = list()
        for i, (start, length) in enumerate(holes):
            # i=2
            # start = holes[i][0]
            # length = holes[i][1]
            if length < self.smoothing_window:  # smoothing (rolling) handled this hole
                continue
            else:
                start = start + (self.smoothing_window - 1)
                length = length - (self.smoothing_window - 1)
                newHoles.append((start, length))
        for i, (start, length) in enumerate(newHoles):
            # i = 2
            # start = newHoles[i][0]
            # length = newHoles[i][1]
            data_processed.iloc[start: start + length] = np.nan
        return data_processed

    def smoothing_process(self, data, missingValuesProcess):
        # fill NaN with mean of  values around the hole
        holes = missingValuesProcess.elementOccurrences(Occurrences=np.isnan(data.values).all(axis=1).astype(np.int),
                                                        element=1)  # find the new holes after the 1th missing values process
        # NaN_indxs = data.iloc[:,0].index[data.iloc[:,0].apply(np.isnan)]
        for col in data.columns:
            # col = 'Roll'  # Pitch Yaw Roll
            data_col_processed = self.smoothing(data.loc[:, col].copy(), holes)
            if data.columns[0] == col:  # at the first iteration there is no processed data yet
                processedData = data_col_processed.copy()
                continue
            processedData = np.column_stack((processedData, data_col_processed))
        processedData = pd.DataFrame(processedData, columns=data.columns)
        return processedData

    def manipulate(self, data):
        processedData = None
        # the first interpolation - methos: mix, order:3, maxHole=5
        missingValuesProcess = PreProc_missingValues(min_valid=5, max_hole_size=5,
                                                     facialExp=self.facialExp)  # 1th missing values process
        data, holes = missingValuesProcess.manipulate(data)

        # fill NaN with mean of  values around the hole
        holes = missingValuesProcess.elementOccurrences(Occurrences=np.isnan(data.values).all(axis=1).astype(np.int),
                                                        element=1)  # find the new holes after the 1th missing values process
        # NaN_indxs = data.iloc[:,0].index[data.iloc[:,0].apply(np.isnan)]
        for col in data.columns:
            # col = 'Pitch'  # Pitch Yaw Roll
            data_col_processed = self.smoothing(data.loc[:, col], holes)
            # seconed missing values process
            missingValuesProcess = PreProc_missingValues(min_valid=5, max_hole_size=5, facialExp=self.facialExp)
            data_col_processed, _ = missingValuesProcess.manipulate(data_col_processed)
            # PROBLEMS!!@#!$ 1@#@#
            # data_col_processed = data_col_processed.to_numpy()
            if data.columns[0] == col:  # at the first iteration there is no processed data yet
                processedData = data_col_processed
                continue
            # processedData = np.column_stack((processedData, smoothingData))
            processedData = np.column_stack((processedData, data_col_processed))
            processedData = pd.DataFrame(processedData, columns=data.columns)
        return pd.DataFrame(processedData)

    def plotGraphComparison(self, FrameIndex, originData, ProcessedData, features, movieID, participantID, title=None):
        # if self.facialExp != "head":
        # originData[originData == 0] = np.nan
        # ProcessedData[ProcessedData == 0] = np.nan
        for f in features:
            if f == "FrameIndex":
                continue
            # f='Pitch'
            plt.figure()
            plt.plot(FrameIndex.values, originData[f].values, marker='', color='red', linestyle='-',
                     label="Original Data")
            plt.plot(FrameIndex.values, ProcessedData[f].values, marker='', color='blue', linestyle='-',
                     label="Processed Data")
            plt.legend()
            plt.ylabel("{:s}".format(f))
            plt.xlabel("Frame Index")
            plt.title("'{:s}': Movie '{:s}', Participant '{:s}'".format(title, movieID, participantID))
            plt.show()

    def specific_data_preprocess(self, movie, subject, features, first_max_hole_size=6, second_max_hole_size=6,
                                 is_second_time=False, is_only_smooth=True, is_origin_data_after_smooth=False):
        if self.facialExp == "head":
            preProc_missingValues_instance = PreProc_missingValues(min_valid=5, max_hole_size=first_max_hole_size,
                                                                   facialExp=self.facialExp)
            origin_data, ProcessedData = preProc_missingValues_instance.missingValues_process(movie, subject,
                                                                                              isSaving=False,
                                                                                              isPlot=False)
            if is_second_time:
                # smotthing process
                ProcessedData = self.smoothing_process(ProcessedData[features].copy(), preProc_missingValues_instance)
                if is_origin_data_after_smooth:
                    origin_data = ProcessedData.copy()
                # second interpolation
                if not is_only_smooth:
                    missingValuesProcess = PreProc_missingValues(min_valid=5, max_hole_size=second_max_hole_size,
                                                                 facialExp=self.facialExp)
                    ProcessedData, _ = missingValuesProcess.manipulate(ProcessedData[features].copy())
        else:
            preProc_missingValues_instance = PreProc_missingValues(min_valid=2, max_hole_size=1,
                                                                   facialExp=self.facialExp)
            origin_data, ProcessedData = preProc_missingValues_instance.missingValues_process(movie, subject,
                                                                                              isSaving=False,
                                                                                              isPlot=False)
            preProc_missingValues_instance = PreProc_missingValues(min_valid=5, max_hole_size=first_max_hole_size,
                                                                   facialExp=self.facialExp)
            ProcessedData, _ = preProc_missingValues_instance.manipulate(ProcessedData.copy())
        return origin_data, ProcessedData

    def plot_raw_data(self, movie, subject, originDir):
        # originDir=PreProc.data2PreProcess_dir
        mid_dir = os.path.join(originDir, movie)
        sraw = os.path.join(mid_dir, subject + ".txt")
        features = self.chosenCols[self.facialExp]
        frameIndex = ["FrameIndex"]
        originData = pd.read_csv(sraw, sep="\t", usecols=features + frameIndex)
        frameIndexs = originData["FrameIndex"]
        for f in features:
            plt.figure()
            plt.plot(frameIndexs.values, originData[f].values, marker='', color='red', linestyle='-')
            plt.ylabel("{:s}".format(f))
            plt.xlabel("Frame Index")
            plt.title("{:s}: Movie {:s}, Participant {:s}".format(self.facialExp, movie, subject))
            plt.show()

    def inspect_process(self, movie, subject, originDir, first_max_hole_size=6, second_max_hole_size=6, isPlot=True,
                        is_second_time=True, is_only_smooth=True, title=None, is_origin_on_processed_data_plot=False,
                        is_origin_data_after_smooth=True):
        # originDir=PreProc.data2PreProcess_dir
        mid_dir = os.path.join(originDir, movie)
        sraw = os.path.join(mid_dir, subject + ".txt")
        features = self.chosenCols[self.facialExp]
        frameIndex = ["FrameIndex"]
        originData = pd.read_csv(sraw, sep="\t", usecols=features + frameIndex)
        frameIndexs = originData["FrameIndex"]
        # data = originData[features].copy()
        origin_data, ProcessedData = self.specific_data_preprocess(movie, subject, features,
                                                                   first_max_hole_size=first_max_hole_size,
                                                                   second_max_hole_size=second_max_hole_size,
                                                                   is_second_time=is_second_time,
                                                                   is_only_smooth=is_only_smooth,
                                                                   is_origin_data_after_smooth=is_origin_data_after_smooth)
        if isPlot:
            if is_origin_on_processed_data_plot:
                PreProc_missingValues(facialExp=self.facialExp).plotGraphComparison(frameIndexs, origin_data,
                                                                                    ProcessedData, features, movie,
                                                                                    subject, title=title)
            else:
                self.plotGraphComparison(frameIndexs, originData, ProcessedData, features, movie, subject, title=title)
        return originData, ProcessedData

    def get_outliers(self, data_2_boundaries, data_2_outliers, num_std_2_boundaries=3):
        # data_2_outliers = new_data
        # data_2_boundaries = origin_data
        # calculate summary statistics
        data_mean, data_std = np.nanmean(data_2_boundaries), np.nanstd(data_2_boundaries)
        # identify outliers
        cut_off = data_std * num_std_2_boundaries
        lower, upper = data_mean - cut_off, data_mean + cut_off
        # identify outliers
        outliers = data_2_outliers[(data_2_outliers < lower) | (data_2_outliers > upper)].values.tolist()
        # outliers = [x for x in data_2_outliers if x < lower or x > upper]
        return outliers

    def is_problem_new_data(self, origin_data, new_data):
        # new_data = ProcessedData_feature
        # origin_data = origin_data_feature
        # a = ProcessedData_feature.dropna()
        # b = origin_data_feature.dropna()
        old_outliers = self.get_outliers(data_2_boundaries=origin_data, data_2_outliers=origin_data)
        new_outliers = self.get_outliers(data_2_boundaries=origin_data, data_2_outliers=new_data)
        diff_outliers = list(set(new_outliers) - set(old_outliers))
        is_probelm_in_new_data = 0
        if len(diff_outliers) != 0:
            for outlier in diff_outliers:
                # outlier = diff_outliers[0]
                new_data = new_data.reset_index(drop=True)
                outlier_index = new_data[new_data == outlier].index[0]
                start_sub_section = outlier_index - 10
                end_sub_section = outlier_index + 10
                if start_sub_section < 0:
                    start_sub_section = 0
                if end_sub_section > new_data.size:
                    end_sub_section = new_data.size
                sub_section = new_data[start_sub_section: end_sub_section]
                if any(item in sub_section.values.tolist() for item in old_outliers):
                    break
                sub_section_outliers = self.get_outliers(sub_section, sub_section, num_std_2_boundaries=1)
                if outlier in sub_section_outliers:
                    is_probelm_in_new_data = 1
                    break
        return is_probelm_in_new_data

    def get_report_process_completing_values(self, min_hole_size=2, max_hole_size=8, first_max_hole_size=6,
                                             is_second_time=False, is_only_smooth=True,
                                             is_origin_data_after_smooth=False):
        hole_range = range(min_hole_size, max_hole_size, 1)
        movies_list = self.mids
        features = self.chosenCols[self.facialExp]
        result_outliers_data = pd.DataFrame([], columns=["Movie", "Participant", "hole_size", "feature",
                                                         "is_probelm_in_new_data", "max_hole",
                                                         "max_hole_divided_totals_rows", "% rows with NaN values"])
        for hole in hole_range:
            # hole=2
            print("****" + str(hole) + "****")
            for mid in movies_list:
                # mid="m1"
                print(mid)
                mid_dir = os.path.join(PreProc.data2PreProcess_dir, mid)
                subjids = sorted(
                    list(set([re.findall(PreProc.fpat, f)[0] for f in glob.glob(os.path.join(mid_dir, "*.txt"))])))
                for s in subjids:
                    # s='200'
                    print(s)
                    originData = pd.read_csv(os.path.join(mid_dir, s + ".txt"), sep="\t")
                    originData_byFeatures = originData[features]
                    preProc_missingValues_instance = PreProc_missingValues(max_hole_size=hole, facialExp=self.facialExp)
                    max_hole, percentage_max_hole, precentage_rows_any_NaN = preProc_missingValues_instance.get_max_hole_and_precentage_rows_any_NaN(
                        originData_byFeatures)
                    if is_second_time:
                        first_max_hole_size = first_max_hole_size
                        second_max_hole_size = hole
                    else:
                        first_max_hole_size = hole
                        second_max_hole_size = hole
                    origin_data, ProcessedData = self.specific_data_preprocess(mid, s, features,
                                                                               first_max_hole_size=first_max_hole_size,
                                                                               second_max_hole_size=second_max_hole_size,
                                                                               is_second_time=is_second_time,
                                                                               is_only_smooth=is_only_smooth,
                                                                               is_origin_data_after_smooth=is_origin_data_after_smooth)
                    if is_origin_data_after_smooth:
                        originData_byFeatures = origin_data[features]
                    for col in originData_byFeatures.columns:
                        if (max_hole > 10) & (self.facialExp != "head"):
                            is_probelm_in_new_data = -1
                        else:
                            # col = originData_byFeatures.columns[2]
                            origin_data_feature = originData_byFeatures[col]
                            ProcessedData_feature = ProcessedData[col]
                            is_probelm_in_new_data = self.is_problem_new_data(origin_data_feature,
                                                                              ProcessedData_feature)
                        new_row = pd.DataFrame(
                            [(mid, s, hole, col, is_probelm_in_new_data, max_hole, percentage_max_hole,
                              precentage_rows_any_NaN)],
                            columns=["Movie", "Participant", "hole_size", "feature", "is_probelm_in_new_data",
                                     "max_hole", "max_hole_divided_totals_rows", "% rows with NaN values"])
                        result_outliers_data = result_outliers_data.append(new_row)
        result_outliers_data.to_csv("{}_report_process_completing_values.csv".format(self.facialExp), index=False)
        return result_outliers_data

    def chooseWindowSize_smoothing(self, originDir=data2PreProcess_dir, minWindow=2, maxWindow=10):
        SAD = {}
        MSE = {}
        windows2check = range(minWindow, maxWindow + 1)
        for window in windows2check:
            # window=2
            print("################" + str(window) + "################")
            p = PreProc(smoothing_window=window, maxValid=5, facialExp=self.facialExp)
            SAD_w = {}
            MSE_w = {}
            features = p.chosenCols[p.facialExp]
            for mid in p.mids:
                # mid = 'm11'
                print("preprocessing {:s}".format(mid))
                mid_dir = os.path.join(p.data2PreProcess_dir, mid)
                subjids = sorted(
                    list(set([re.findall(p.fpat, f)[0] for f in glob.glob(os.path.join(mid_dir, "*.txt"))])))
                if len(subjids) == 0:
                    continue
                if not os.path.isdir(mid_dir):
                    print("can't find raw data dir for {:s} at {:s}. skipping.\n".format(mid, mid_dir))
                    continue
                for s in subjids:
                    # s='228'
                    print(s)
                    originData, processedData = p.inspect_process(movie=mid, subject=s, originDir=originDir,
                                                                  is_only_smooth=True, isPlot=False)
                    originData = originData[features]
                    NaN_indexs = originData.iloc[:, 0].index[originData.iloc[:, 0].apply(np.isnan)]
                    originData = originData.drop(NaN_indexs)
                    processedData = processedData.drop(NaN_indexs)
                    for f in features:
                        if f not in SAD_w:
                            SAD_w[f] = []
                            MSE_w[f] = []
                        SAD_w[f].append(np.sum(abs(originData[f] - processedData[f])))
                        MSE_w[f].append(np.sum((originData[f] - processedData[f]) ** 2))
            for f in features:
                if f not in SAD:
                    SAD[f] = []
                    MSE[f] = []
                SAD[f].append(np.mean(SAD_w[f]))
                MSE[f].append(np.mean(MSE_w[f]))

        for f in features:
            plt.figure()
            plt.ylabel("{:s}".format(f))
            plt.xlabel("window size")
            plt.title("Sum of absolute differences for different window sizes")
            plt.plot(windows2check, SAD[f])
            plt.grid()
            plt.show()

            plt.figure()
            plt.ylabel("{:s}".format(f))
            plt.xlabel("window size")
            plt.title("Mean of square  differences for different window sizes")
            plt.plot(windows2check, MSE[f])
            plt.grid()
            plt.show()
        return SAD, MSE

    def get_movie_dir(self, mid, log):
        print("preprocessing {:s}".format(mid))
        mid_dir = "None"
        noDir_msg = "can't find raw data dir for {:s} at {:s}. skipping.\n".format(mid, mid_dir)
        noSubjects_msg = "can't find any subjects for {:s} at {:s}. skipping.\n".format(mid, mid_dir)
        mid_dir_rawData = os.path.join(PreProc.rawDatadir, mid)
        if os.path.isdir(mid_dir_rawData):
            subjids = sorted(
                list(set([re.findall(PreProc.fpat, f)[0] for f in glob.glob(os.path.join(mid_dir_rawData, "*.txt"))])))
            if len(subjids) == 0:
                mid_dir = os.path.join(PreProc.data2PreProcess_dir, mid)
                if not os.path.isdir(mid_dir):
                    log.write(noDir_msg)
                    return None, None
                subjids = sorted(
                    list(set([re.findall(PreProc.fpat, f)[0] for f in glob.glob(os.path.join(mid_dir, "*.txt"))])))
                if len(subjids) == 0:
                    log.write(noSubjects_msg)
                    return None, None
            else:
                mid_dir = mid_dir_rawData
        else:
            mid_dir = os.path.join(PreProc.data2PreProcess_dir, mid)
            if not os.path.isdir(mid_dir):
                log.write(noDir_msg)
                return None, None
            subjids = sorted(
                list(set([re.findall(PreProc.fpat, f)[0] for f in glob.glob(os.path.join(mid_dir, "*.txt"))])))
            if len(subjids) == 0:
                log.write(noSubjects_msg)
                return None, None
        log.write("entring {:s}\n".format(mid))
        return mid_dir, subjids, log

    def cutDataByMovieSubject(self, data, mid, s, log):
        log.write("cutting {:s}".format(s))
        timing2split_data = pd.read_excel(
            "C:\\Users\\Coral\\facial head behavior and road hazard perceptions\\data\\timing2split.xlsx",
            sheet_name="screenRecored_affectivaTime")
        movie_times = timing2split_data.loc[:, ["participant", "part", mid]]
        start = movie_times[(movie_times["participant"] == int(s)) & (
                    movie_times["part"] == "affectiva_start")]  # the start point for this movie of this participant
        end = movie_times[(movie_times["participant"] == int(s)) & (
                    movie_times["part"] == "affectiva_end")]  # the end point for this movie of this participant
        cuttedData = data[
            (data['MediaTime'] > int(start[mid])) & (data['MediaTime'] < int(end[mid]))]  # cutting the data
        return cuttedData, log

    def unifyingLengths(self, DB, mid, subjids, lengths, processed_withNaN, processed_noNaN, features, log):
        log.write("unifying lengths of {:s}".format(mid))
        minLength = lengths.min()
        for s in subjids:
            if lengths[s] == minLength:
                data = pd.DataFrame(DB[s][()], columns=features)
            else:  # lengths[s] > minLength
                data = pd.DataFrame(DB[s][()], columns=features)
                data = data.iloc[np.arange(0, minLength),]
            processed_withNaN.create_dataset(s, data=data[features])
            n_NaN = np.sum(data.iloc[:, 1].isnull())
            if n_NaN == 0:
                processed_noNaN.create_dataset(s, data=data)
        return processed_withNaN, processed_noNaN, log, minLength

    def preprocess(self, first_max_hole_size=3, second_max_hole_size=6, is_second_time=True, is_only_smooth=True,
                   isSaving=False):
        # p = PreProc(facialExp="facial") # to DEL
        log = io.StringIO()
        hd5dir = os.path.join(PreProc.datadir, "{}_DB.hd5".format(self.facialExp))
        hd = h5py.File(hd5dir, "a")
        features = self.chosenCols[self.facialExp]
        atts = {
            'smoothing_window': self.smoothing_window,
            'first_max_hole_size': first_max_hole_size,
            'second_max_hole_size': second_max_hole_size,
            'fps': PreProc.fps,
            'source_lengths': None,
            'length': 0,
            'facialExp': self.facialExp
        }
        for mid in self.mids:
            # mid = 'm1'
            mid_dir, subjids, log = self.get_movie_dir(mid, log)
            if (subjids == None) | (mid_dir == None):
                continue
            # DB handeling
            if mid in hd:
                if "base" in hd[mid]:
                    del hd[mid]["base"]
                rawData = hd[mid].create_group("base")
                if "processed_data_no_NaN" in hd[mid]:
                    del hd[mid]["processed_data_no_NaN"]
                processed_noNaN = hd[mid].create_group("processed_data_no_NaN")
                if "processed_data_with_NaN" in hd[mid]:
                    del hd[mid]["processed_data_with_NaN"]
                processed_withNaN = hd[mid].create_group("processed_data_with_NaN")
                if "UnevenDataLengths" in hd[mid]:
                    del hd[mid]["UnevenDataLengths"]
                processed_UnevenDataLengths = hd[mid].create_group("UnevenDataLengths")
            else:
                rawData = hd.create_group(mid + "/base")
                processed_noNaN = hd.create_group(mid + "/processed_data_no_NaN")
                processed_withNaN = hd.create_group(mid + "/processed_data_with_NaN")
                processed_UnevenDataLengths = hd.create_group(mid + "/UnevenDataLengths")
            lengths = pd.Series(index=subjids, dtype='float64')
            for s in subjids:
                # s='233'
                sraw = os.path.join(mid_dir, s + ".txt")
                if not os.path.isfile(sraw):
                    log.write("\tcan't find raw data file for subject {:s}. at {:s}. skipping.\n".format(s, sraw))
                    continue
                log.write("preprocessing {:s}".format(s))
                # preprocessing
                origin_data, ProcessedData = self.specific_data_preprocess(mid, s, features,
                                                                           first_max_hole_size=first_max_hole_size,
                                                                           second_max_hole_size=second_max_hole_size,
                                                                           is_second_time=is_second_time,
                                                                           is_only_smooth=is_only_smooth)
                data_2_cut = origin_data.copy()
                data_2_cut[features] = ProcessedData[features]
                features2DB = ["MediaTime"] + features
                # cut exactly by movie length
                processed_cuttedData, log = self.cutDataByMovieSubject(data_2_cut.copy(), mid, s, log)
                origin_cuttedData, log = self.cutDataByMovieSubject(origin_data.copy(), mid, s, log)
                rawData.create_dataset(s, data=origin_cuttedData[features2DB].copy())
                atts["features"] = origin_cuttedData[features2DB].columns.tolist()
                if isSaving:
                    r = RawDataCreation()
                    data_2_save = origin_data.copy()
                    data_2_save[features] = ProcessedData[features]
                    r.saveDataByMovies(data_2_save, mid, s, "rawDatadir")
                processed_UnevenDataLengths.create_dataset(s, data=processed_cuttedData[features2DB])
                lengths[s] = processed_cuttedData.shape[0]
            processed_withNaN, processed_noNaN, log, minLength = self.unifyingLengths(processed_UnevenDataLengths, mid,
                                                                                      subjids, lengths,
                                                                                      processed_withNaN,
                                                                                      processed_noNaN, features2DB, log)
            print("updating meta data")
            atts['source_lengths'] = lengths
            atts['length'] = minLength
            for key, value in atts.items():
                hd[mid].attrs.create(key, value)
        hd.close()
        print("done")
        return log

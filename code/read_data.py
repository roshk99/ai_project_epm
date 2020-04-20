import xlrd
import numpy as np
import datetime
import pickle
import os

def read_final_grades_sheet(sheet, student_ids):
    initial_headers = sheet.row_values(0)
    question_headers = []
    point_values = []
    for header in initial_headers[1:]:
        split_header = header.split('\n')
        question_headers.append(split_header[0])
        point_values.append(split_header[1].split(' ')[0][1:])
    point_values = np.array(point_values)

    values = np.zeros((student_ids.shape[0], sheet.ncols-1))
    for row_ind in range(1, sheet.nrows):
        row = sheet.row_values(row_ind)
        id = int(row[0])
        values[id,:] = row[1:]

    return question_headers, point_values, values

def read_final_grades(student_ids):
    wb = xlrd.open_workbook('../EPM_Dataset/Data/final_grades.xlsx')

    attempt1 = read_final_grades_sheet(wb.sheet_by_index(0), student_ids)
    attempt2 = read_final_grades_sheet(wb.sheet_by_index(1), student_ids)
    return attempt1[0], attempt1[1], attempt1[2], attempt2[2]

def import_logs():
    with open('../EPM_Dataset/Data/logs.txt') as f:
        content = f.readlines()
        headers = content[0].split('\t')
        headers[-1] = headers[-1][:-1]

        student_ids = []
        values = []
        for row in content[1:]:
            row = row.split('\t')
            row[-1] = row[-1][:-1]
            student_ids.append(row[0])
            values.append(row[1:])

        student_ids = np.array(student_ids).astype('int')
        values = np.array(values).astype('int')

        return student_ids, values

def read_int_grades():
    wb = xlrd.open_workbook('../EPM_Dataset/Data/intermediate_grades.xlsx')
    sheet = wb.sheet_by_index(0)
    headers = sheet.row_values(0)[1:]

    values = np.zeros((sheet.nrows-1, len(headers)))
    for row_ind in range(1, sheet.nrows):
        values[row_ind-1,:] = sheet.row_values(row_ind)[1:]

    return values

def read_exercises(student_ids, logs):
    headers = ['Session Number', 'Student Id', 'Exercise', 'Activity',
        'Start_time', 'End_time', 'Idle Time', 'Mouse Wheel', 'Mouse Wheel Click',
        'Mouse Wheel Click Left', 'Mouse Wheel Click Right', 'Mouse Movement', 'Keystroke']
    foldername = '../EPM_Dataset/Data/Processes'
    sessions = np.arange(1,7)

    time0 = datetime.datetime.strptime('02.10.2014 01:00:00', '%m.%d.%Y %H:%M:%S')

    T = []
    X = []
    for id in student_ids:
        T.append([])
        X.append([])

    for session in sessions:
        for id_num in np.arange(logs.shape[0]):
            if logs[id_num,session-1]:
                filename = '{}/Session {}/{}'.format(foldername, session, student_ids[id_num])
                try:
                    #print(filename)
                    with open(filename) as f:
                        content = f.readlines()
                        tt = np.empty((len(content)))
                        xx = np.empty((len(content), 8))
                        for row_ind, row in enumerate(content):
                            split_row = row.split(', ')

                            exercise, activity, start_time, end_time, idle_time, \
                                mouse_wheel, click, click_left, click_right, mouse_move, \
                                keystroke = split_row[2:]

                            #Ignore exercise and activity for now
                            if not start_time[1] == '.':
                                total_time = 0.0
                                start_time = None
                            else:
                                start_time = datetime.datetime.strptime(start_time, '%m.%d.%Y %H:%M:%S')
                                start_time = (start_time - time0).total_seconds()
                                end_time = datetime.datetime.strptime(end_time, '%m.%d.%Y %H:%M:%S')
                                end_time = (end_time - time0).total_seconds()
                                total_time = end_time - start_time

                            idle_time = float(idle_time)/1000
                            mouse_wheel = float(mouse_wheel)
                            click = float(click)
                            click_left = float(click_left)
                            click_right = float(click_right)
                            mouse_move = float(mouse_move)
                            keystroke = float(keystroke)

                            tt[row_ind] = start_time
                            xx[row_ind,:] = [total_time, idle_time, mouse_wheel, click, \
                                click_left, click_right, mouse_move, keystroke]
                except:
                    print('Could not open', filename)
                T[id_num].append(tt)
                X[id_num].append(xx)

    return T, X

def import_data(categories=2):

    if not 'data.pkl' in os.listdir():
        student_ids, logs = import_logs()
        questions, points, final_grades_1, final_grades_2 = read_final_grades(student_ids)
        int_grades = read_int_grades()
        time, features = read_exercises(student_ids, logs)
        grades = np.mean(np.array([final_grades_1[:,-1], final_grades_2[:,-1]]),axis=0)

        my_data = {'grades': grades, 'features': features}
        output = open('data.pkl', 'wb')
        pickle.dump(my_data, output)
        output.close()
    else:
        with open('data.pkl', 'rb') as f:
            data = pickle.load(f)
        grades, features = data['grades'], data['features']

    bins = np.linspace(np.min(grades), np.max(grades), categories+1)
    Y = np.zeros_like(grades).astype('int')
    for label, (left, right) in enumerate(zip(bins[0:-1], bins[1:])):
        Y[np.where((grades >= left) & (grades <= right))[0]] = label
    Y = Y.tolist()

    y = [[element] for element in Y]    ### We need a list of lists

    X = []
    for cur_feat in features:
        X.append(np.vstack(cur_feat).tolist())

    return X, y


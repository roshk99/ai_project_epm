import xlrd
import numpy as np

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

        student_ids = np.array(student_ids)
        values = np.array(values)

        return student_ids, values

def read_int_grades():
    wb = xlrd.open_workbook('../EPM_Dataset/Data/intermediate_grades.xlsx')
    sheet = wb.sheet_by_index(0)
    headers = sheet.row_values(0)[1:]

    values = np.zeros((sheet.nrows-1, len(headers)))
    for row_ind in range(1, sheet.nrows):
        values[row_ind-1,:] = sheet.row_values(row_ind)[1:]

    return values

def main():
    student_ids, logs = import_logs()
    questions, points, final_grades_1, final_grades_2 = read_final_grades(student_ids)
    int_grades = read_int_grades()

if __name__ == '__main__':
    main()

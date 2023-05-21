import cv2
import os
import pydicom

inputdir = '/home/jaehan0605/MAIN_ASDfundus/dicoms/'
outdir = '/home/jaehan0605/MAIN_ASDfundus/new_asd_images/'

test_list = [ f for f in  os.listdir(inputdir)]

for f in test_list:   # remove "[:10]" to convert all images 
    ds = pydicom.read_file(inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if f.split('_')[4] == 'ASD':
        if f[-3:]=='dcm':
            cv2.imwrite(outdir + f.replace('.dcm','.png'),img) # write png image
        elif f[-3:]=='DCM':
            cv2.imwrite(outdir + f.replace('.DCM','.png'),img) # write png image
        else:
            print("---------------------------------------------------------------")
    else:
        pass



# ###### 매칭 위해서 number ~ 환자 번호 매칭하기
# import pandas as pd
# import os
# import pydicom
# direction = '/home/jaehan0605/MAIN_ASDfundus/dicoms_ASD/'
# test_list = [f for f in  os.listdir(direction)]

# matching_list = []
# for i in test_list:
#     id = i[2:6]
#     patient_id = pydicom.dcmread(f'/home/jaehan0605/MAIN_ASDfundus/dicoms_ASD/{i}').PatientID
#     new_list = [id, patient_id]
#     matching_list.append(new_list)

# matching_df = pd.DataFrame(matching_list, columns=['number','patient_id']).drop_duplicates()
# len(matching_df)
# matching_df.to_csv('/home/jaehan0605/MAIN_ASDfundus/matching_df.csv')

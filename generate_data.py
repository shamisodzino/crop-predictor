import csv
import random

def generate_crop_data(crop_name):
    soil_ph = round(random.uniform(6.0, 7.0), 1)
    ##rainfall = random.uniform(1000, 1500) Tobbaco
    rainfall = random.uniform(900, 1300)
    mn = random.uniform(20, 60)
    zn = random.uniform(3, 10)
    cu = random.uniform(2, 5)
    p = random.uniform(15, 30)
    k = random.uniform(150, 300)
    n = random.uniform(150, 300)
    temp = random.uniform(15, 30)
    soil_drainage = random.choice(["Well-drained"])

    return [rainfall, temp, soil_ph,  n,p,k, mn, zn, cu, soil_drainage, crop_name]

def append_crop_data_to_csv(crop_name, file_name="crop_data.csv"):
    crops = []
    for i in range(350):
        crops.append(generate_crop_data(crop_name))

    with open(file_name, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in crops:
            csv_writer.writerow(row)

if __name__ == "__main__":
    crop_name = input("Enter the crop name: ")
    append_crop_data_to_csv(crop_name)
    print("Crop data has been appended to the CSV file.")

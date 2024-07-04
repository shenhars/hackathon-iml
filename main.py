import predict_passenger_boarding
import predict_trip_duration

# main.py --training_set ./data/HU.BER/train_bus_schedule.csv --test_set ./data/HU.BER/train_bus_schedule.csv --out ./data/results/output1.csv --train True --model_type base
"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH --train True/False --model_type base/rf/gb

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv --train True --model_type rf

"""

if __name__ == '__main__':
    print("TASK 1")
    predict_passenger_boarding.main()
    print("TASK 2")
    predict_trip_duration.main()

from data_processing import UserInput
from data_processing import DataProcessor












if __name__ == "__main__":

    print("Welcome to the model training app...")
    
    # Instantiate UserInput and run_all method
    user_input_instance = UserInput()
    user_input_instance.run_all()

    # Instantiate DataProcessor with data from UserInput and run_all method
    data_processor_instance = DataProcessor(user_input_instance.data.copy(), user_input_instance.file_path, user_input_instance.ml_type, user_input_instance.target_column)

    data_processor_instance.run_all()

    
    
from data_processing import UserInput
from data_processing import DataProcessor












if __name__ == "__main__":
    # Instantiate UserInput and run_all method
    user_input_instance = UserInput()
    user_input_instance.run_all()

    # Instantiate DataProcessor with data from UserInput and run_all method
    data_processor_instance = DataProcessor(user_input_instance.data.copy(), user_input_instance.file_path)
    data_processor_instance.run_all()

    # Print the head of both the original and modified data to check the changes
    print("Original Data:")
    print(user_input_instance.data.head())

    print("\nModified Data:")
    print(data_processor_instance.data.head())
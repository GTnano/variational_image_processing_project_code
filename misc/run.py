if __name__ == '__main__':
    models_std = [5, 10, 15, 20 ,25]
    tests_std = [5, 10, 15, 20 ,25]

    for model_std in models_std:
        for test_std in tests_std:
            with open("test_t-step_denoisers.py") as file:
                exec(file.read(), {'arg1': model_std, 'arg2': test_std})
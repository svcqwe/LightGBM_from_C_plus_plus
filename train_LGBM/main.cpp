#include <LightGBM/c_api.h>
#include <vector>
#include <iostream>
#include <windows.h>
#include <fstream>
#include <sstream>
using namespace std;


// Функция для загрузки CSV в память (строки как vector<double>, label в колонке 0, фичи с 1)
vector<vector<double>> load_csv(const char* filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    string line;
    bool header_skipped = false;
    while (getline(file, line)) {
        if (!header_skipped) { header_skipped = true; continue; }  // Пропуск header
        stringstream ss(line);
        string token;
        vector<double> row;
        while (getline(ss, token, ',')) {
            try { row.push_back(stod(token)); } catch (...) { row.push_back(0.0); }  // NaN как 0
        }
        data.push_back(row);
    }
    return data;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);

    // Загрузка данных в память для split
    const char* filename = "Dataset_LGBM.csv";
    vector<vector<double>> raw_data = load_csv(filename);
    int nrows = raw_data.size();
    if (nrows == 0) { cerr << "Пустой датасет!" << endl; return -1; }
    int num_features = raw_data[0].size() - 1;  // Label в колонке 0, фичи с 1
    int train_size = nrows * 0.8;  // 80% train, 20% valid 

    // Flatten train data и labels (row-major)
    vector<double> train_data;
    vector<float> train_labels(train_size);
    for (int i = 0; i < train_size; ++i) {
        auto& row = raw_data[i];
        train_labels[i] = static_cast<float>(row[0]);  // Label
        train_data.insert(train_data.end(), row.begin() + 1, row.end());  // Фичи
    }

    // Flatten valid data и labels
    int valid_size = nrows - train_size;
    vector<double> valid_data;
    vector<float> valid_labels(valid_size);
    for (int i = train_size; i < nrows; ++i) {
        auto& row = raw_data[i];
        valid_labels[i - train_size] = static_cast<float>(row[0]);
        valid_data.insert(valid_data.end(), row.begin() + 1, row.end());
    }

    // Создание train_dataset
    DatasetHandle train_dataset;
    const char* dataset_params = "min_data_in_bin=1 min_data_in_leaf=1";  // Низкие для малого датасета
    int result_train_dataset = LGBM_DatasetCreateFromMat(train_data.data(), C_API_DTYPE_FLOAT64, train_size, num_features, 1,
                                       dataset_params, nullptr, &train_dataset);
    if (result_train_dataset != 0) { cerr << "Train датасет ошибка: " << LGBM_GetLastError() << endl; return -1; }
    LGBM_DatasetSetField(train_dataset, "label", train_labels.data(), train_size, C_API_DTYPE_FLOAT32);

    // Создание valid_dataset с reference=train
    DatasetHandle valid_dataset;
    int result_valid_dataset = LGBM_DatasetCreateFromMat(valid_data.data(), C_API_DTYPE_FLOAT64, valid_size, num_features, 1,
                                       dataset_params, train_dataset, &valid_dataset);
    if (result_valid_dataset != 0) { cerr << "Valid датасет ошибка: " << LGBM_GetLastError() << endl; return -1; }
    LGBM_DatasetSetField(valid_dataset, "label", valid_labels.data(), valid_size, C_API_DTYPE_FLOAT32);

    cout << "Датасеты созданы! Train: " << train_size << ", Valid: " << valid_size << endl;

    // Создание Booster
    BoosterHandle booster;
    const char* booster_params = "objective=binary "
                                 "metric=accuracy,auc,binary_logloss "
                                 "max_depth=7 "
                                 "scale_pos_weight=3.2 "
                                 "learning_rate=0.1 "
                                 "num_iterations=100 "
                                 "seed=42 "
                                 "min_data_in_leaf=1 "
                                 "force_row_wise=true "
                                 "num_leaves=6 "
                                 "feature_fraction=0.8 "
                                 "lambda_l1=0.1 lambda_l2=0.1 "
                                 "verbose=2 "  // 1 для метрик, без debug
                                 "early_stopping_round=10";  // Стоп при no improvement на valid
    int result_booster_create = LGBM_BoosterCreate(train_dataset, booster_params, &booster);
    if (result_booster_create != 0) { cerr << "Booster ошибка: " << LGBM_GetLastError() << endl; return -1; }

    // Добавление valid
    int result_add_valid_data = LGBM_BoosterAddValidData(booster, valid_dataset);
    if (result_add_valid_data != 0) { cerr << "Valid добавление ошибка: " << LGBM_GetLastError() << endl; return -1; }

    // Обучение с выводом метрик
    int is_finished = 0;
    int num_metrics;
    LGBM_BoosterGetEvalCounts(booster, &num_metrics);
    vector<double> eval_results(num_metrics * 2);  // Для train (0) и valid (1), double для API
    for (int i = 0; i < 100 && is_finished == 0; ++i) {
        int result_booster_iter = LGBM_BoosterUpdateOneIter(booster, &is_finished);
        if (result_booster_iter != 0) { cerr << "Итерация ошибка " << i << ": " << LGBM_GetLastError() << endl; break; }

        // Вывод метрик для train
        LGBM_BoosterGetEval(booster, 0, &num_metrics, eval_results.data());
        cout << "Итерация " << i << " Train: accuracy=" << eval_results[0] << ", auc=" << eval_results[1] << ", logloss=" << eval_results[2] << endl;

        // Вывод метрик для valid
        LGBM_BoosterGetEval(booster, 1, &num_metrics, eval_results.data());
        cout << "Итерация " << i << " Valid: accuracy=" << eval_results[0] << ", auc=" << eval_results[1] << ", logloss=" << eval_results[2] << endl;
    }

    cout << "Модель обучена!" << endl;

    LGBM_BoosterSaveModel(booster, 0, -1, 0, "LGBM_model.txt");

    LGBM_DatasetFree(train_dataset);
    LGBM_DatasetFree(valid_dataset);
    LGBM_BoosterFree(booster);

    return 0;
}
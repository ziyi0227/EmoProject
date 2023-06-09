#include "emo.h"
#include <cmath>

// 准备训练集和测试集数据
vector<map<int, double>> train_data, test_data, emo_data; // 使用map实现稀疏矩阵
vector<int> train_label, test_label, emo_label;

ifstream train_file(R"(D:\clion\EmoProject\train_emtion.txt)");
ifstream test_file(R"(D:\clion\EmoProject\test_emtion.txt)");
ifstream emo_file(R"(D:\clion\EmoProject\emo_emtion.txt)");

string line;

int main() {
    //读取训练集数据
    while (getline(train_file, line)) {
        istringstream iss(line);
        double value;
        map<int, double> sample; //用于存储情绪特征，维度从1-62
        int index = 1;
        for (int i = 0; i < 62; i++) {
            iss >> value;
            sample[index] = value;
            index++;
        }
        int label;
        iss >> label;
        train_data.push_back(sample); //情绪特征存储在train_data中
        train_label.push_back(label); //情绪情况存储在train_data中
    }
    train_file.close();

    //读取测试集数据
    while (getline(test_file, line)) {
        istringstream iss(line);
        double value;
        map<int, double> sample; //用于存储情绪特征，维度1-62
        int index = 1;
        for (int i = 0; i < 62; i++) {
            iss >> value;
            sample[index] = value;
            index++;
        }
        int label;
        iss >> label;
        test_data.push_back(sample); //情绪特征存储在test_data中
        test_label.push_back(label); //情绪情况存储在test_data中
    }
    test_file.close();

    //将稀疏矩阵转换为LIBSVM格式
    svm_problem train_prob, test_prob, emo_prob;
    train_prob.l = train_data.size(); //这里的l为L,存储训练集的样本数
    train_prob.y = new double[train_prob.l]; //存储训练数据的标签，是一个double类型的数组，长度为train_prob.l
    train_prob.x = new svm_node*[train_prob.l]; //存储训练数据的特征矩阵，它是一个长度为train_prob.l的svm_node指针数组

    for (int i = 0; i < train_prob.l; i++) {
        train_prob.y[i] = train_label[i];
        train_prob.x[i] = new svm_node[train_data[i].size() + 1]; //在 libsvm 库中，每个特征向量必须以一个空节点结尾。
        int j = 0;
        for (auto it = train_data[i].begin(); it != train_data[i].end(); it++) {
            train_prob.x[i][j].index = it->first;
            train_prob.x[i][j].value = it->second;
            j++;
        }
        train_prob.x[i][j].index = -1; //在libsvm库中，每个特征向量都必须以一个空节点结尾，这是因为libsvm库在内部使用稀疏矩阵的数据结构来处理特征向量，而空节点的索引值为-1，表示特征向量的结束。
    }

    test_prob.l = test_data.size();
    test_prob.y = new double[test_prob.l];
    test_prob.x = new svm_node*[test_prob.l];
    for (int i = 0; i < test_prob.l; i++) {
        test_prob.y[i] = test_label[i];
        test_prob.x[i] = new svm_node[test_data[i].size() + 1];
        int j = 0;
        for (auto it =test_data[i].begin(); it != test_data[i].end(); it++) {
            test_prob.x[i][j].index = it->first;
            test_prob.x[i][j].value = it->second;
            j++;
        }
        test_prob.x[i][j].index = -1;
    }

    //训练模型

    //LIBSVM是一种流行的支持向量机（SVM）模型训练软件包。
    //param.svm_type是LIBSVM库中使用的一个参数，该库是一种流行的支持向量机（SVM）模型训练软件包。
    svm_parameter param{};//SVM参数
    param.svm_type = C_SVC;//param.svm_type是LIBSVM库中使用的一个参数。
    // C-SVC：这是用于分类问题的标准SVM，其目标是找到将数据分成两个类的超平面。参数C控制最大化间隔和最小化分类误差之间的权衡。
    param.kernel_type = RBF;
    // 该参数指定SVM使用的内核类型，这里设定为径向基函数（RBF）内核。
    // RBF内核是一种常用的SVM内核函数，它可以将数据映射到高维空间，并计算样本之间的相似度。
    // 使用RBF内核的SVM可以有效地处理非线性分类问题，因为它可以捕捉到数据在高维空间中的非线性结构，从而提高分类的准确性。
    param.gamma = 0.06;
    // 控制着支持向量机在高维空间中对数据进行映射时，不同样本之间的相似度程度。
    // 具体来说，gamma值越大，相似度下降得越快，决策边界就越复杂，模型的训练误差会减小，但过度拟合的风险也就越大。
    // 通常情况下，gamma的取值范围在0到1之间。如果数据集很大，可以尝试使用较小的gamma值，以便更好地处理噪声和不必要的特征。
    // 相反，如果数据集较小并且具有明显的特征，可以尝试使用较大的gamma值，以获得更好的分类效果。
    param.C = 1;
    // 该参数是SVM中的一个正则化参数，控制着SVM学习分类边界的平滑度和正确分类的权重。
    // 具体来说，C值越大，SVM的分类边界越复杂，对训练集的拟合程度越高，但过度拟合的风险也就越大。

    //设置权重条件，用于处理类别不平衡问题
    int num_positive_labels = count(train_label.begin(), train_label.end(), 1);
    int num_negative_labels = count(train_label.begin(), train_label.end(), -1);
    double positive_weight = 1.0 / num_positive_labels;
    double negative_weight = 1.0 / num_negative_labels;
    double weights[2] = {positive_weight, negative_weight};
    param.weight_label = new int[2] {1, -1};
    param.weight = new double[2] {positive_weight, negative_weight};

    printf("进入训练");
    svm_problem* prob_ptr = &train_prob;
    svm_check_parameter(prob_ptr, &param); // 检查参数是否合法
    svm_model* model = svm_train(prob_ptr, &param);
    printf("训练完成\n");
    printf("-----------------------------------------------------------------------------------------------");
    printf("进入测试\n");

    //测试模型
    int correct = 0;
    for (int i = 0; i < test_prob.l; i++) {
        double pred_label = svm_predict(model, test_prob.x[i]); // 对测试样本进行分类
        if (pred_label == test_prob.y[i]) {
            correct++;
        }
    }
    double accuracy = (double)correct *100.0/ test_prob.l; // 计算分类准确率
    cout << "Accuracy = " << accuracy <<"%"<< endl;

    /**
     * optimization finished, #iter = 10000000
     * nu = 0.054406
     * obj = -6.220485, rho = 0.240120
     * nSV = 28, nBSV = 3
     * Total nSV = 95
     * 进入测试Accuracy = 100%

     * 这段文本表明您可能运行了一个优化算法，可能是支持向量机（SVM）的优化算法，最大迭代次数设置为1000万次。
     * 该算法已经收敛，最终步长或学习率为0.054406，目标函数值为-6.220485。参数rho的值为0.240120，表示决策边界相对于支持向量的位置。
     * 发现的支持向量数为28，边界支持向量数（即具有非零拉格朗日乘子的支持向量）为3。总支持向量数，包括不在边界上的支持向量，为95。
     */

    /*查询系统1.0*/
    //读取文件
    while (getline(emo_file, line)) {
        istringstream iss(line);
        double value;
        map<int, double> sample; //用于存储情绪特征，维度从1-62
        int index = 1;
        for (int i = 0; i < 62; i++) {
            iss >> value;
            sample[index] = value;
            index++;
        }
        // int label;
        // iss >> label;
        emo_data.push_back(sample); //情绪特征存储在train_data中
        // train_label.push_back(label); //情绪情况存储在train_data中
    }
    emo_file.close();

    //转换格式
    emo_prob.l = emo_data.size();
    emo_prob.x = new svm_node*[emo_prob.l];
    for (int i = 0; i < emo_prob.l; i++) {
        emo_prob.x[i] = new svm_node[emo_data[i].size() + 1];
        int j = 0;
        for (auto it =emo_data[i].begin(); it != emo_data[i].end(); it++) {
            emo_prob.x[i][j].index = it->first;
            emo_prob.x[i][j].value = it->second;
            j++;
        }
        emo_prob.x[i][j].index = -1;
    }

    //结果输出
    int Count0 = 0, Count1 = 0, Count2 = 0;
    double rate0 = 0, rate1 = 0, rate2 = 0;
    for (int i = 0; i < emo_prob.l; i++) {
        double pred_label = svm_predict(model, emo_prob.x[i]); // 对测试样本进行分类
        printf("Emo sample %d: Predicted label = %g\n", i, pred_label); // 打印预测结果和真实标签
        if (pred_label == 0) { // 统计预测结果为0的数量
            Count0++;
        } else if (pred_label == 1) { // 统计预测结果为1的数量
            Count1++;
        } else if (pred_label == 2) { // 统计预测结果为1的数量
            Count2++;
        }
    }
    printf("打印统计结果\n");
    printf(" 0 samples: %d个\n 1 samples: %d个\n 2 samples: %d个\n", Count0, Count1, Count2); // 打印统计结果
    int Count = Count0 + Count1 + Count2;
    rate0 = Count0 *100.0/ Count;
    rate1 = Count1 *100.0/ Count;
    rate2 = Count2 *100.0/ Count;
    printf("打印比率");
    printf("0 rates: %lf%, 1 rates: %lf%, 1 rates: %lf%\n", rate0, rate1, rate2); //打印比率

//    /*查询系统2.0*/
//    //窗口
//    MainWindow::MainWindow(QWidget *parent)
//    : QMainWindow(parent)
//            , ui(new Ui::MainWindow)
//    {
//        ui->setupUi(this);
//
//        QVBoxLayout *layout = new QVBoxLayout;
//
//        m_headerLabel = new QLabel("Emotion Classifier", this);
//        m_loadFileButton = new QPushButton("Load Data File", this);
//        m_outputTextEdit = new QPlainTextEdit(this);
//        m_outputTextEdit->setReadOnly(true);
//
//        layout->addWidget(m_headerLabel);
//        layout->addWidget(m_loadFileButton);
//        layout->addWidget(m_outputTextEdit);
//
//        QWidget *centralWidget = new QWidget(this);
//        centralWidget->setLayout(layout);
//        setCentralWidget(centralWidget);
//
//        connect(m_loadFileButton, &QPushButton::clicked, this, &MainWindow::on_loadFileButton_clicked);
//    }
//
//    MainWindow::~MainWindow()
//    {
//        delete ui;
//    }
//
//    void MainWindow::on_loadFileButton_clicked()
//    {
//        QString emoFilePath = QFileDialog::getOpenFileName(this, tr("Open Emotion File"), "", tr("Emotion Files (*.txt);;All Files (*)"));
//
//        if (!emoFilePath.isEmpty()) {
//            processData(emoFilePath.toStdString());
//            analyzeData();
//        }
//    }
//
//    std::string MainWindow::loadEmoFile()
//    {
//        return QFileDialog::getOpenFileName(this, tr("Open Emotion File"), "", tr("Emotion Files (*.txt);;All Files (*)")).toStdString();
//    }
//
//    void MainWindow::processData(std::string emoFilePath)
//    {
//        // 读取文件
//        while (getline(emo_file, line)) {
//            istringstream iss(line);
//            double value;
//            map<int, double> sample; //用于存储情绪特征，维度从1-62
//            int index = 1;
//            for (int i = 0; i < 62; i++) {
//                iss >> value;
//                sample[index] = value;
//                index++;
//            }
//  //        int label;
//  //        iss >> label;
//            emo_data.push_back(sample); //情绪特征存储在train_data中
//  //        train_label.push_back(label); //情绪情况存储在train_data中
//        }
//        emo_file.close();
//
//        //转换格式
//        emo_prob.l = emo_data.size();
//        emo_prob.x = new svm_node*[emo_prob.l];
//        for (int i = 0; i < emo_prob.l; i++) {
//            emo_prob.x[i] = new svm_node[emo_data[i].size() + 1];
//            int j = 0;
//            for (auto it =emo_data[i].begin(); it != emo_data[i].end(); it++) {
//                emo_prob.x[i][j].index = it->first;
//                emo_prob.x[i][j].value = it->second;
//                j++;
//            }
//            emo_prob.x[i][j].index = -1;
//        }
//    }
//
//    void MainWindow::analyzeData()
//    {
//        // 结果输出
//        int Count0 = 0, Count1 = 0, Count2 = 0;
//        double rate0 = 0, rate1 = 0, rate2 = 0;
//        // 与您的代码相同，但将printf替换为将结果追加到m_outputTextEdit中
//        m_outputTextEdit->clear();
//        for (int i = 0; i < test_prob.l; i++) {
//            double pred_label = svm_predict(model, emo_prob.x[i]);
//            QString output = QString("Emo sample %1: Predicted label = %2\n").arg(i).arg(pred_label);
//            m_outputTextEdit->appendPlainText(output);
//            if (pred_label == 0) { // 统计预测结果为0的数量
//                Count0++;
//            } else if (pred_label == 1) { // 统计预测结果为1的数量
//                Count1++;
//            } else if (pred_label == 2) { // 统计预测结果为1的数量
//                Count2++;
//            }
//        }
//
//        QString summary = QString("0 samples: %1, 1 samples: %2, 2 samples: %3\n").arg(Count0).arg(Count1).arg(Count2);
//        m_outputTextEdit->appendPlainText(summary);
//
//        int Count = Count0 + Count1 + Count2;
//        rate0 = Count0 / Count;
//        rate1 = Count1 / Count;
//        rate2 = Count2 / Count;
//        QString rates = QString("0 rates: %1, 1 rates: %2, 1 rates: %3\n").arg(rate0).arg(rate1).arg(rate2);
//        m_outputTextEdit->appendPlainText(rates);
//    }


    printf("释放内存");

    // 释放内存
    svm_free_and_destroy_model(&model);
    delete[] param.weight_label;
    delete[] param.weight;
    for (int i = 0; i < train_prob.l; i++) {
        delete[] train_prob.x[i];
    }
    delete[] train_prob.x;
    delete[] train_prob.y;
    for (int i = 0; i < test_prob.l; i++) {
        delete[] test_prob.x[i];
    }
    delete[] test_prob.x;
    delete[] test_prob.y;
    for (int i = 0; i < emo_prob.l; i++) {
        delete[] emo_prob.x[i];
    }
    delete[] emo_prob.x;

    return 0;
}
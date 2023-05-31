//
// Created by 86150 on 2023-05-28.
//

#ifndef EMOPROJECTCLION_EMO_H
#define EMOPROJECTCLION_EMO_H

//#include <QMainWindow>
//#include <QLabel>
//#include <QPushButton>
//#include <QVBoxLayout>
//#include <QFileDialog>
//#include <QTextEdit>
//#include <QPlainTextEdit>

#include <iostream>
#include <fstream>
#include <bits/stdc++.h>
#include "svm.h" // LIBSVM头文件
using namespace std;

//QT_BEGIN_NAMESPACE
//namespace Ui { class MainWindow; }
//QT_END_NAMESPACE
//
//class MainWindow : public QMainWindow
//{
//    Q_OBJECT
//
//public:
//    explicit MainWindow(QWidget *parent = nullptr);
//    ~MainWindow();
//
//private slots:
//            void on_loadFileButton_clicked();
//    void processAndAnalyzeData();
//
//private:
//    Ui::MainWindow *ui;
//    QLabel *m_headerLabel;
//    QPushButton *m_loadFileButton;
//    QPlainTextEdit *m_outputTextEdit;
//
//    std::vector<std::map<int, double>> emo_data;
//    struct svm_problem emo_prob;
//    struct svm_model *model;
//    std::string loadEmoFile();
//    void processData(std::string emoFilePath);
//    void analyzeData();
//};

#endif //EMOPROJECTCLION_EMO_H

本项目基于《LogBERT: Log Anomaly Detection via BERT》改进
实验运行：
    cd (HDFS|BGL|TBird)
    构建词汇表：python logbert.py vocab
    训练模型：python logbert.py train
    测试模型：python logbert.py predict
项目说明：
    日志异常检测是指对计算机系统或应用程序生成的日志数据进行分析和监控，以便检测出其中的异常行为或事件。
    在计算机系统中，日志记录了各种活动，包括用户登录、文件访问、系统错误等。通过分析这些日志数据，可以发现异常模式
    或行为，如潜在的安全威胁、系统故障、性能问题等。该项目主要用于日志异常检测，通过掩码预测与构建超球体（
    超球体（也称为球形或球形异常检测器）是一种基于距离的异常检测方法，其核心思想是将数据点映射到一个高维空间中，
    并在该空间中使用球体来表示正常数据的区域。异常数据点通常远离这个球体，因此可以被识别为异常。）双任务结合进一步提高异常检测的性能。

改进说明：源项目在掩码预测中，采用的是静态掩码的方式，为了进一步提高模型的泛化能力，本实验采用动态掩码的方式。
    具体而言，对于原始项目来说，其只经过一次掩码，而本实验通过每个epoch，都重新的对数据进行随机掩码，相当于
    原始项目只有n个掩码的数据，而通过动态掩码，本次实验逻辑上获得了n*epoch个掩码的数据，泛化能力更好。

日志数据集：output/bgl/train output/bgl/test_normal output/bgl/test_abnormal
        output/hdfs/train output/hdfs/test_normal output/hdfs/test_abnormal

实验环境：
    Ubuntu 20.04
    NVIDIA driver 460.73.01
    CUDA 11.2
    Python 3.8
    PyTorch 1.9.0
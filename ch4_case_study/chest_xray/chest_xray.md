# 肺炎检测

本案例使用卷积神经网络建立分类系统，该系统根据胸部X射线检测来检查可疑。由于训练图像已正确标记以显示哪些患者被诊断出患有肺炎，因此有利于我们使用监督学习。过程中将介绍图像增广（image augmentation）技术通过对训练图像做改动，比如压缩，灰度化，旋转等}来提高模型的验证和分类准确性。

# 问题定义：

如今，肺炎的风险对于许多人来说是巨大的。特别是在发展中国家，数十亿人面临能源短缺，并长期依赖着一些会造成环境污染的资源。世界卫生组织估计，每年有超过400万人死于与家庭空气污染有关的疾病，这其中就包括肺炎。

随着医学成像技术和设备的进步，借助X线成像、超声波成像、磁共振成像等，为医生提供越来越多样且精确的信息。海量的数据搭配革故鼎新的深度学习技术，使其能够在医疗领域取得突破性进展。其中，智能化运用在肺炎检测中占有非常重要的地位。

本案例目的是为用户提供网络应用程序，用户只需将肺部X射线图像上传到浏览器，随后部署的模型对其进行分析预测。最后，返回结果，结果则是判断该患者是否存在肺炎迹象。

## 开发环境
我们鼓励读者为该案例研究建立一个独立的虚拟环境，从而避免项目和使用框架的版本不一致而产生的错误以及其他不必要的麻烦。我们需要使用四个模块：keras，numpy，tensorflow 和 flask。

ch5_Python中提供了有关如何使用这四个模块的详细介绍。在这里，我们提供了有关如何为该案例研究创建虚拟环境的分步指南。我们在requirements.txt中提供了所需Python库的列表。我们可以简单地在终端中输入以下命令进行安装：
```sh
pip install -r requirements.txt
```
## 项目运行
*  [训练模型] - 运行 model.py，训练后的模型将自动保存到 model 的文件夹下，并命名为 model_cnn.h5 
* [运行服务器] - 执行 server.py 与Web服务器交互。如果一切顺利，它将返回一个您本地主机的服务器地址。
```sh
* Serving Flask app "server" (lazy loading)
* Environment: production
WARNING: This is a development server. Do not use it in a production deployment.
Use a production WSGI server instead.
* Debug mode: off
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
* [使用Web 应用程序] - 打开浏览器，然后输入网站地址：(http://127.0.0.1:5000/)。此时，你的浏览器将显示一个简单的网站应用程序
* [停止服务器] - 按CTRL+C
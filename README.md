# 可爱可倾的博客

使用 GitHub pages 托管的静态博客

## 使用的框架

1. [Hexo](https://github.com/hexojs/hexo)
2. [anzhiyu主题](https://github.com/anzhiyu-c/hexo-theme-anzhiyu)

### 部署命令

1. 安装npm包 `npm install`
2. 本地预览 `hexo clean && hexo generate && hexo bangumi -u && hexo swpp && hexo server`
3. 部署github `hexo clean && hexo generate && hexo bangumi -u && hexo swpp && hexo deploy`

## 未来计划

1. 评论系统 Twikoo/Giscus
2. 托管图片-几个方案-博客里面
3. CDN以及mathjax CDN
4. 自动部署
5. Live2D

## 已知问题

1. favicon图片修改

### 第三方插件bug

1. 主题bug: page页面top_img空白

## 版本历史

|  修订时间   |       修订内容         | 版本号 |
| :--------: | :------------------: | :----: |
| 2024-08-30 |    创建博客并正式上线   |  v0.0  |
| 2024-10-30 |        更换主题       |  v0.1  |

### 更新日志

#### v0.0

1. (240830)创建hexo博客
2. (240830)使用yun主题
3. (240830)修改图片尺寸显示格式，适配本地Markdown编辑器

#### v0.1

1. (241025)修复图片路径错误不显示的问题
2. (241026)初步更换主题为anzhiyu
3. (241026)修复布局比例失当的问题
4. (241030)修复归档等界面图片布局错乱的问题，适配适配本地Markdown编辑器
5. (241030)修复因为路径配置导致的部分图片不显示的问题
6. (241030)删除page页面top_img，原主题有bug

#### v0.2

1. (241031)配置公式系统，可复制公式
2. (241031)全面采用pandoc渲染
3. (241031)适配图片渲染格式，扩展性增强
4. (241031)加入搜索功能
5. (241031)加入隐私政策
6. (241031)加入51A统计系统
7. (241031)加入关于页面

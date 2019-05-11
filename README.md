# ZJUSCT Blog - Source Branch

<img align="right" width="159px" src="https://raw.githubusercontent.com/fish98/fish98.github.io/master/2018/05/17/stack/thumbnail.png">

<p style="color:white;font-size:30px;text-align:center;background-color:red;">Do not merge this to master!</p>

This is the source branch of official ZJUSCT blog repo.

Anyone can write your Markdown pages and share cool ideas here! But please get a fork first(Detail's below).

## Usage

On using [Hexo](https://hexo.io/) to generate DHTML static Website from Markdown files, the `source` branch is just a "source code" project from which Hexo generates real Website Project(to [`master`](https://github.com/ZJUSCT/ZJUSCT.github.io/tree/master) branch). We must consider of Multi-user conflict and the publishing right.

So, the probable way to contribute to the `source` site is:

1. Fork this branch to your own.
2. Edit & Preview & Commit on your own fork.
3. On finishing, make a **New Pull Request** return to this branch. And wait Administrator to check and publish.

### On Clone your fork to local

+ After `git clone` you can run `npm install` to install necessary dependencies

+ run `hexo g` to generate your markdown file into static page

For more detailed usages please check [The Usage for Hexo](https://hexo.io/zh-cn/docs/index.html)

### If you're Admin

On receving pull request, check it and accept. Then pull the main branch and use `hexo g` and `hexo d`, then the page can be published.

---

## More

### where are the posts

They are lying in /source

The normal post are in dir _post

each Page owns a seperate directory

### About ejs Configuration

If you have any idea about css config changes for this simple intro website. You can modefy the config files in /themes/fish/layout by yourself and assign a pull request or simply contact TTfish.

Or if you are inetested in writing a new theme specially for ZJUSCT, please do not hesitate. We will be fully grateful!

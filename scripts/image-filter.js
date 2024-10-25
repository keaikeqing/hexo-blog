hexo.extend.filter.register("after_render:html", function (htmlContent) {
  // 替换 img 标签，添加自定义的样式和标题
  return htmlContent.replace(
    /<img src="(.*?)" alt="(.*?)" title="(.*?)"(.*?)>/g,
    (match, src, alt, title, rest) => {
      let width, height;
      const sizeMatch = src.match(/#(\d+)x(\d+)/);

      if (sizeMatch) {
        width = sizeMatch[1];
        height = sizeMatch[2];
        src = src.replace(/#(\d+)x(\d+)/, "");
      }

      return `
          <div style="text-align: center;">
              <img src="${src}" alt="${alt}" ${rest} 
                  ${
                    width && height
                      ? `style="display: block; margin: 0 auto; width: ${width}%; height: ${height}%; max-width: 500px; max-height: 500px;"`
                      : 'style="display: block; margin: 0 auto; max-width: 500px; max-height: 500px;"'
                  }>
              <span style="margin-top: 10px; text-decoration: underline; text-underline-offset: 2px; text-decoration-color: #d9d9d9; font-size: 20px; display: block;">${title}</span>
          </div>
      `;
    }
  );
});

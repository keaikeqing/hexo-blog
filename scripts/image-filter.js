hexo.extend.filter.register("after_render:html", function (htmlContent) {
  // 匹配 figure 标签并提取其中内容
  return htmlContent.replace(
    /<figure[^>]*>([\s\S]*?)<\/figure>/g,
    (figureMatch) => {
      // 提取 img 标签和 figcaption 标签
      const imgMatch = figureMatch.match(/<img[^>]*>/);
      const captionMatch = figureMatch.match(
        /<figcaption[^>]*>(.*?)<\/figcaption>/
      );

      if (!imgMatch) return figureMatch; // 如果没有 img 标签，保持原样

      // 解析图片标签中的属性
      const srcMatch = imgMatch[0].match(/src\s*=\s*["'](.*?)["']/);
      const altMatch = imgMatch[0].match(/alt\s*=\s*["'](.*?)["']/);
      const titleMatch = imgMatch[0].match(/title\s*=\s*["'](.*?)["']/);
      const lazySrcMatch = imgMatch[0].match(
        /data-lazy-src\s*=\s*["'](.*?)["']/
      );

      // 获取实际使用的 src（优先使用 data-lazy-src）
      let src = lazySrcMatch ? lazySrcMatch[1] : srcMatch ? srcMatch[1] : "";
      const alt = altMatch ? altMatch[1] : "";
      const title = titleMatch ? titleMatch[1] : "";
      const caption = captionMatch ? captionMatch[1] : ""; // 获取标题

      // 解析尺寸
      const sizeMatch = src.match(/#(\d+)x(\d+)/);
      const errorHandler =
        "onerror=\"this.onerror=null,this.src='/assets/404.webp'\"";
      src = src.replace(/#(\d+)x(\d+)/, "");

      if (sizeMatch) {
        const width = sizeMatch[1];
        const height = sizeMatch[2];

        return `
        <div style="text-align: center;">
            <img 
                src="${src}"
                alt="${alt}"
                title="${title}"
                ${errorHandler}
                data-lazy-src="${src}"
                ${
                  width && height
                    ? `style="display: block; margin: 0 auto; width: ${width}%; height: ${height}%; max-width: 500px; max-height: 500px;"`
                    : 'style="display: block; margin: 0 auto; max-width: 500px; max-height: 500px;"'
                }
            >
            ${
              title
                ? `<span style="margin-top: 10px; text-decoration: underline; text-underline-offset: 2px; text-decoration-color: #d9d9d9; font-size: 20px; display: block;">${title}</span>`
                : caption
                ? `<span style="margin-top: 10px; text-decoration: underline; text-underline-offset: 2px; text-decoration-color: #d9d9d9; font-size: 20px; display: block;">${caption}</span>`
                : ""
            }          
        </div>
      `;
      } else {
        return `
        <div style="text-align: center;">
            <img 
                src="${src}"
                alt="${alt}"
                title="${title}"
                ${errorHandler}
                data-lazy-src="${src}"
                ${'style="display: block; margin: 0 auto; max-width: 500px; max-height: 500px;"'}
            >
            ${
              title
                ? `<span style="margin-top: 10px; text-decoration: underline; text-underline-offset: 2px; text-decoration-color: #d9d9d9; font-size: 20px; display: block;">${title}</span>`
                : caption
                ? `<span style="margin-top: 10px; text-decoration: underline; text-underline-offset: 2px; text-decoration-color: #d9d9d9; font-size: 20px; display: block;">${caption}</span>`
                : ""
            }          
        </div>
      `;
      }
    }
  );
});

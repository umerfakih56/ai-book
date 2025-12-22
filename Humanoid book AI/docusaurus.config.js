const config = {
  title: "Physical AI & Humanoid Robotics",
  tagline: "Comprehensive guide to building intelligent humanoid robots",
  url: "http://localhost:3000",  // Using localhost for development
  baseUrl: "/",
  favicon: "img/favicon.ico",
  organizationName: "umerfakih56",
  projectName: "Physical-AI-Humanoid-Robotics-Course-Book",

  i18n: {
    defaultLocale: "en",
    locales: ["en", "ur"],
    localeConfigs: {
      en: {
        label: "English",
        direction: "ltr"
      },
      ur: {
        label: "اردو (Urdu)",
        direction: "rtl"
      }
    }
  },

  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "warn",

  themeConfig: {
    navbar: {
      title: "Physical AI & Humanoid Robotics",
      items: [
        { 
          type: "doc", 
          docId: "intro", 
          position: "left", 
          label: "Course" 
        },
        { 
          to: "https://github.com/umerfakih56/ai-book", 
          label: "GitHub", 
          position: "right" 
        },
        {
          type: 'dropdown',
          label: '🌐 Language',
          position: 'right',
          items: [
            {
              label: 'English',
              to: '?locale=en',
              activeBaseRegex: '\\/?$|\\/?\\?locale=en'
            },
            {
              label: 'اردو',
              to: '?locale=ur',
              activeBaseRegex: '\\/?\\?locale=ur'
            }
          ]
        }
      ]
    },
    footer: {
      style: 'dark',
      copyright: `Copyright ${new Date().getFullYear()} Physical AI & Humanoid Robotics. Built with Docusaurus.`,
    },
 
  },

  presets: [
    [
      "@docusaurus/preset-classic",
      {
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          editUrl: "https://github.com/umerfakih56/ai-book",
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      },
    ],
  ],

  plugins: [
    function translateButtonPlugin() {
      return {
        name: 'translate-button-plugin',
        injectHtmlTags() {
          return {
            headTags: [
              {
                tagName: 'link',
                rel: 'stylesheet',
                href: 'https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic&display=swap'
              }
            ],
            postBodyTags: [
                {
                tagName: 'script',
                innerHTML: `
                  (function(){
                    const FONT_FAMILY = 'Noto Naskh Arabic, serif';

                    function normalizeUrl(u){
                      try{ return new URL(u, window.location.origin); }catch(e){ return null; }
                    }

                    function getLocaleFromUrl(url){
                      const u = normalizeUrl(url || window.location.href);
                      if(!u) return null;
                      const p = u.searchParams.get('locale');
                      if(p) return p;
                      // check path prefix like /ur/
                      const m = u.pathname.match(/^\/(..)(?:\/|$)/);
                      if(m) return m[1];
                      return null;
                    }

                    function applyRTL(){
                      try{
                        document.documentElement.setAttribute('dir','rtl');
                        document.documentElement.lang = 'ur';
                        document.body.style.direction = 'rtl';
                        document.body.style.fontFamily = FONT_FAMILY;
                        document.body.classList.add('locale-ur');
                      }catch(e){console.error('applyRTL error', e)}
                    }

                    function removeRTL(){
                      try{
                        document.documentElement.removeAttribute('dir');
                        document.documentElement.lang = 'en';
                        document.body.style.direction = '';
                        document.body.style.fontFamily = '';
                        document.body.classList.remove('locale-ur');
                      }catch(e){console.error('removeRTL error', e)}
                    }

                    function applyLocaleFromUrl(url){
                      const locale = getLocaleFromUrl(url);
                      if(locale === 'ur') applyRTL(); else removeRTL();
                    }

                    // SPA navigation: wrap pushState/replaceState and listen to popstate
                    (function(){
                      const _push = history.pushState;
                      history.pushState = function(){ const r = _push.apply(this, arguments); setTimeout(()=>applyLocaleFromUrl(location.href),50); return r; };
                      const _replace = history.replaceState;
                      history.replaceState = function(){ const r = _replace.apply(this, arguments); setTimeout(()=>applyLocaleFromUrl(location.href),50); return r; };
                      window.addEventListener('popstate', ()=> setTimeout(()=>applyLocaleFromUrl(location.href),50));
                    })();

                    // Attach handlers to locale links and re-run detection when DOM changes
                    function attachLocaleHandlers(){
                      const selectors = ['a[href*="?locale=ur"]','a[href*="/ur/"]','a[data-locale="ur"]','.dropdown__link[href*="locale=ur"]'];
                      const nodes = document.querySelectorAll(selectors.join(','));
                      nodes.forEach(n=>{
                        if(n.__localeHandler) return;
                        n.__localeHandler = true;
                        n.addEventListener('click', function(){
                          // allow navigation, but apply locale after navigation completes
                          setTimeout(()=>applyLocaleFromUrl(location.href), 100);
                        });
                      });
                    }

                    const mo = new MutationObserver(()=> attachLocaleHandlers());
                    mo.observe(document.documentElement || document.body, { childList:true, subtree:true });

                    // initial run
                    document.addEventListener('DOMContentLoaded', function(){
                      attachLocaleHandlers();
                      applyLocaleFromUrl(location.href);

                      // Also support a simple translate flow when ?translate=true is present
                      try{
                        const urlp = new URLSearchParams(window.location.search);
                        if(urlp.get('translate') === 'true'){
                          setTimeout(async ()=>{
                            const article = document.querySelector('article') || document.querySelector('.markdown') || document.querySelector('[class*="content"]') || document.body;
                            const contentText = article ? article.innerText : document.body.innerText;
                            if(!contentText || contentText.length < 20){ console.warn('No content found to translate'); return; }
                            try{
                              const resp = await fetch('/api/translate', {
                                method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ text: contentText })
                              });
                              if(resp.ok){ const j = await resp.json(); if(j.translation){
                                document.body.innerHTML = '<div style="padding: 20px; max-width: 900px; margin: 0 auto; min-height: 100vh; direction: rtl; font-family: '+FONT_FAMILY+';"><div style="text-align: right; font-size: 18px; line-height: 1.9;">'+ j.translation +'</div><div style="margin-top: 40px; text-align: center; position: fixed; bottom: 20px; left: 0; right: 0;"><button onclick="window.history.back()" style="padding: 10px 20px; background: #2563eb; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px;">Back to English</button></div></div>';
                              } else console.error('No translation in response', j); } else { console.error('Translate API failed', resp.status); }
                            }catch(e){ console.error('Translate request failed', e); }
                          }, 500);
                        }
                      }catch(e){console.error(e)}
                    });
                  })();
                `,
              },
            ],
          };
        },
      };
    }
  ]
};

module.exports = config;
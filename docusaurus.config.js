const config = {
  title: "Physical AI & Humanoid Robotics",
  tagline: "Comprehensive guide to building intelligent humanoid robots",
  url: "http://localhost:3000",  // Using localhost for development
  baseUrl: "/",
  favicon: "img/favicon.ico",
  organizationName: "umer-fakih56",
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
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: false,
    },
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
          label: 'Language',
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
          editUrl: "https://github.com/umerfakih56/ai-book/edit/main/",
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
                  document.addEventListener('DOMContentLoaded', function() {
                    console.log('Translation script loaded');
                    
                    // Check if we should show the translation
                    const urlParams = new URLSearchParams(window.location.search);
                    const shouldTranslate = urlParams.get('translate') === 'true';
                    console.log('Should translate:', shouldTranslate);
                    console.log('Current URL:', window.location.href);
                    
                    if (shouldTranslate) {
                      console.log('Starting translation process...');
                      // Wait a bit for content to load
                      setTimeout(() => {
                        const text = document.body.innerText;
                        console.log('Text to translate (length):', text.length);
                        console.log('First 100 chars:', text.substring(0, 100));
                        
                        if (text.length === 0) {
                          console.log('No text found, trying alternative selectors...');
                          // Try to get content from main article area
                          const article = document.querySelector('article') || document.querySelector('.markdown') || document.querySelector('[class*="content"]');
                          const contentText = article ? article.innerText : document.body.innerText;
                          console.log('Content text length:', contentText.length);
                          console.log('Content first 100 chars:', contentText.substring(0, 100));
                          
                          if (contentText.length > 0) {
                            translateContent(contentText);
                          } else {
                            console.error('No content found to translate');
                            alert('No content found to translate');
                          }
                        } else {
                          translateContent(text);
                        }
                      }, 2000); // Wait 2 seconds for content to load
                    }
                    
                    function translateContent(text) {
                      console.log('Translating content:', text.substring(0, 100) + '...');
                      
                      const button = document.querySelector('.navbar__item.dropdown .dropdown__link[href*="locale=ur"]');
                      console.log('Button found:', !!button);
                      
                      if (button) {
                        console.log('Button element:', button);
                        // Show loading state
                        const originalText = button.textContent;
                        button.textContent = 'Translating...';
                        button.style.opacity = '0.7';
                        button.style.cursor = 'wait';
                        
                        console.log('Sending translation request...');
                        fetch('/api/translate', {
                          method: 'POST',
                          headers: {
                            'Content-Type': 'application/json',
                          },
                          body: JSON.stringify({ text }),
                        })
                        .then(response => {
                          console.log('Response status:', response.status);
                          console.log('Response headers:', response.headers);
                          if (!response.ok) {
                            throw new Error('HTTP error! status: ' + response.status);
                          }
                          return response.json();
                        })
                        .then(data => {
                          console.log('Translation response:', data);
                          if (data.translation) {
                            console.log('Translation successful, updating page...');
                            document.body.innerHTML = '<div style="padding: 20px; max-width: 800px; margin: 0 auto; min-height: 100vh; direction: rtl; font-family: Noto Naskh Arabic, serif;"><div style="text-align: right; font-size: 18px; line-height: 2;">' + data.translation + '</div><div style="margin-top: 40px; text-align: center; position: fixed; bottom: 20px; left: 0; right: 0;"><button onclick="window.history.back()" style="padding: 10px 20px; background: #2563eb; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px;">Back to English</button></div></div>';
                          } else {
                            console.error('No translation in response:', data);
                            alert('No translation received');
                          }
                        })
                        .catch(error => {
                          console.error('Translation error:', error);
                          console.error('Error details:', error.message);
                          console.error('Error stack:', error.stack);
                          if (button) {
                            button.textContent = originalText;
                            button.style.opacity = '1';
                            button.style.cursor = 'pointer';
                          }
                          alert('Translation failed. Please try again.');
                        });
                      } else {
                        console.error('Translation button not found');
                        console.log('Available buttons:', document.querySelectorAll('a[href*="locale"]'));
                        console.log('All links:', document.querySelectorAll('a'));
                      }
                    }
                    
                    setTimeout(() => {
                      console.log('Looking for Urdu button...');
                      const urduButton = document.querySelector('a[href*="locale=ur"]');
                      console.log('Primary selector result:', !!urduButton);
                      

                      if (urduButton) {
                        console.log('Urdu button found:', urduButton);
                        urduButton.onclick = function(e) {
                          console.log('Urdu button clicked via onclick!');
                          e.preventDefault();
                          console.log('Event prevented');
                          const currentUrl = new URL(window.location.href);
                          console.log('Current URL before:', currentUrl.toString());
                          currentUrl.searchParams.set('translate', 'true');
                          console.log('New URL with translate:', currentUrl.toString());
                          window.location.href = currentUrl.toString();
                        };
                        urduButton.addEventListener('click', function(e) {
                          console.log('Urdu button clicked via addEventListener!');
                          e.preventDefault();
                          console.log('Event prevented');
                          const currentUrl = new URL(window.location.href);
                          console.log('Current URL before:', currentUrl.toString());
                          currentUrl.searchParams.set('translate', 'true');
                          console.log('New URL with translate:', currentUrl.toString());
                          window.location.href = currentUrl.toString();
                        });
                      } else {
                        console.log('Urdu button not found with primary selector');
                        console.log('Available locale links:', document.querySelectorAll('a[href*="locale"]'));
                        
                        // Try alternative selector
                        const altButton = document.querySelector('.dropdown__link[href*="locale=ur"]');
                        console.log('Alternative selector result:', !!altButton);
                        if (altButton) {
                          console.log('Alternative Urdu button found:', altButton);
                          altButton.addEventListener('click', function(e) {
                            e.preventDefault();
                            console.log('Alternative Urdu button clicked!');
                            const currentUrl = new URL(window.location.href);
                            currentUrl.searchParams.set('translate', 'true');
                            window.location.href = currentUrl.toString();
                          });
                        } else {
                          console.log('No Urdu button found with any selector');
                          console.log('All dropdown links:', document.querySelectorAll('.dropdown__link'));
                          console.log('All links on page:', document.querySelectorAll('a'));
                        }
                      }
                    }, 1000); // Wait 1 second for DOM to be ready
                  });
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
import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', 'bad'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '2e5'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '654'),
            routes: [
              {
                path: '/docs/assessments/',
                component: ComponentCreator('/docs/assessments/', '373'),
                exact: true
              },
              {
                path: '/docs/assessments/gazebo-robot-project',
                component: ComponentCreator('/docs/assessments/gazebo-robot-project', '4a1'),
                exact: true
              },
              {
                path: '/docs/assessments/ros2-basics-quiz',
                component: ComponentCreator('/docs/assessments/ros2-basics-quiz', '22c'),
                exact: true
              },
              {
                path: '/docs/assessments/vla-capstone',
                component: ComponentCreator('/docs/assessments/vla-capstone', '97d'),
                exact: true
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '61d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/learning-paths/',
                component: ComponentCreator('/docs/learning-paths/', '2fa'),
                exact: true
              },
              {
                path: '/docs/learning-paths/complete-physical-ai',
                component: ComponentCreator('/docs/learning-paths/complete-physical-ai', 'bb7'),
                exact: true
              },
              {
                path: '/docs/learning-paths/quick-start',
                component: ComponentCreator('/docs/learning-paths/quick-start', '088'),
                exact: true
              },
              {
                path: '/docs/learning-paths/robotics-foundations',
                component: ComponentCreator('/docs/learning-paths/robotics-foundations', '60c'),
                exact: true
              },
              {
                path: '/docs/learning-paths/simulation-specialist',
                component: ComponentCreator('/docs/learning-paths/simulation-specialist', '695'),
                exact: true
              },
              {
                path: '/docs/learning-paths/software-developer',
                component: ComponentCreator('/docs/learning-paths/software-developer', '112'),
                exact: true
              },
              {
                path: '/docs/module-1/',
                component: ComponentCreator('/docs/module-1/', '3dd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1/ch1-1',
                component: ComponentCreator('/docs/module-1/ch1-1', '9ce'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1/ch1-2',
                component: ComponentCreator('/docs/module-1/ch1-2', '9d9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1/ch1-3',
                component: ComponentCreator('/docs/module-1/ch1-3', 'b0d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1/ch1-4',
                component: ComponentCreator('/docs/module-1/ch1-4', 'a70'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2/',
                component: ComponentCreator('/docs/module-2/', 'ca2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2/ch2-1',
                component: ComponentCreator('/docs/module-2/ch2-1', 'aa0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2/ch2-2',
                component: ComponentCreator('/docs/module-2/ch2-2', '83f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2/ch2-3',
                component: ComponentCreator('/docs/module-2/ch2-3', 'ec2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2/ch2-4',
                component: ComponentCreator('/docs/module-2/ch2-4', 'b4d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3/',
                component: ComponentCreator('/docs/module-3/', '72b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3/ch3-1',
                component: ComponentCreator('/docs/module-3/ch3-1', '166'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3/ch3-2',
                component: ComponentCreator('/docs/module-3/ch3-2', '009'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3/ch3-3',
                component: ComponentCreator('/docs/module-3/ch3-3', '9ac'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3/ch3-4',
                component: ComponentCreator('/docs/module-3/ch3-4', '9ac'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4/',
                component: ComponentCreator('/docs/module-4/', 'c39'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4/ch4-1',
                component: ComponentCreator('/docs/module-4/ch4-1', '8d8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4/ch4-2',
                component: ComponentCreator('/docs/module-4/ch4-2', '025'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4/ch4-3',
                component: ComponentCreator('/docs/module-4/ch4-3', '0a0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4/ch4-4',
                component: ComponentCreator('/docs/module-4/ch4-4', 'daf'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/toc',
                component: ComponentCreator('/docs/toc', '6af'),
                exact: true
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];

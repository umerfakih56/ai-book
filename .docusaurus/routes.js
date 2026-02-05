import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/ur/docs',
    component: ComponentCreator('/ur/docs', '12a'),
    routes: [
      {
        path: '/ur/docs',
        component: ComponentCreator('/ur/docs', '1f0'),
        routes: [
          {
            path: '/ur/docs',
            component: ComponentCreator('/ur/docs', '815'),
            routes: [
              {
                path: '/ur/docs/assessments/',
                component: ComponentCreator('/ur/docs/assessments/', '552'),
                exact: true
              },
              {
                path: '/ur/docs/assessments/gazebo-robot-project',
                component: ComponentCreator('/ur/docs/assessments/gazebo-robot-project', 'af8'),
                exact: true
              },
              {
                path: '/ur/docs/assessments/ros2-basics-quiz',
                component: ComponentCreator('/ur/docs/assessments/ros2-basics-quiz', '419'),
                exact: true
              },
              {
                path: '/ur/docs/assessments/vla-capstone',
                component: ComponentCreator('/ur/docs/assessments/vla-capstone', 'bd5'),
                exact: true
              },
              {
                path: '/ur/docs/intro',
                component: ComponentCreator('/ur/docs/intro', '793'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ur/docs/learning-paths/',
                component: ComponentCreator('/ur/docs/learning-paths/', '11f'),
                exact: true
              },
              {
                path: '/ur/docs/learning-paths/complete-physical-ai',
                component: ComponentCreator('/ur/docs/learning-paths/complete-physical-ai', '5ec'),
                exact: true
              },
              {
                path: '/ur/docs/learning-paths/quick-start',
                component: ComponentCreator('/ur/docs/learning-paths/quick-start', '17a'),
                exact: true
              },
              {
                path: '/ur/docs/learning-paths/robotics-foundations',
                component: ComponentCreator('/ur/docs/learning-paths/robotics-foundations', '751'),
                exact: true
              },
              {
                path: '/ur/docs/learning-paths/simulation-specialist',
                component: ComponentCreator('/ur/docs/learning-paths/simulation-specialist', 'e80'),
                exact: true
              },
              {
                path: '/ur/docs/learning-paths/software-developer',
                component: ComponentCreator('/ur/docs/learning-paths/software-developer', 'ce4'),
                exact: true
              },
              {
                path: '/ur/docs/module-1/',
                component: ComponentCreator('/ur/docs/module-1/', '9b5'),
                exact: true
              },
              {
                path: '/ur/docs/module-1/ch1-1',
                component: ComponentCreator('/ur/docs/module-1/ch1-1', '497'),
                exact: true
              },
              {
                path: '/ur/docs/module-1/ch1-2',
                component: ComponentCreator('/ur/docs/module-1/ch1-2', 'f9a'),
                exact: true
              },
              {
                path: '/ur/docs/module-1/ch1-3',
                component: ComponentCreator('/ur/docs/module-1/ch1-3', '954'),
                exact: true
              },
              {
                path: '/ur/docs/module-1/ch1-4',
                component: ComponentCreator('/ur/docs/module-1/ch1-4', '728'),
                exact: true
              },
              {
                path: '/ur/docs/module-2/',
                component: ComponentCreator('/ur/docs/module-2/', '069'),
                exact: true
              },
              {
                path: '/ur/docs/module-2/ch2-1',
                component: ComponentCreator('/ur/docs/module-2/ch2-1', '3de'),
                exact: true
              },
              {
                path: '/ur/docs/module-2/ch2-2',
                component: ComponentCreator('/ur/docs/module-2/ch2-2', 'c45'),
                exact: true
              },
              {
                path: '/ur/docs/module-2/ch2-3',
                component: ComponentCreator('/ur/docs/module-2/ch2-3', '7a3'),
                exact: true
              },
              {
                path: '/ur/docs/module-2/ch2-4',
                component: ComponentCreator('/ur/docs/module-2/ch2-4', '8d5'),
                exact: true
              },
              {
                path: '/ur/docs/module-3/',
                component: ComponentCreator('/ur/docs/module-3/', '17b'),
                exact: true
              },
              {
                path: '/ur/docs/module-3/ch3-1',
                component: ComponentCreator('/ur/docs/module-3/ch3-1', 'c2b'),
                exact: true
              },
              {
                path: '/ur/docs/module-3/ch3-2',
                component: ComponentCreator('/ur/docs/module-3/ch3-2', '64a'),
                exact: true
              },
              {
                path: '/ur/docs/module-3/ch3-3',
                component: ComponentCreator('/ur/docs/module-3/ch3-3', 'b73'),
                exact: true
              },
              {
                path: '/ur/docs/module-3/ch3-4',
                component: ComponentCreator('/ur/docs/module-3/ch3-4', '685'),
                exact: true
              },
              {
                path: '/ur/docs/module-4/',
                component: ComponentCreator('/ur/docs/module-4/', 'b2c'),
                exact: true
              },
              {
                path: '/ur/docs/module-4/ch4-1',
                component: ComponentCreator('/ur/docs/module-4/ch4-1', '155'),
                exact: true
              },
              {
                path: '/ur/docs/module-4/ch4-2',
                component: ComponentCreator('/ur/docs/module-4/ch4-2', '557'),
                exact: true
              },
              {
                path: '/ur/docs/module-4/ch4-3',
                component: ComponentCreator('/ur/docs/module-4/ch4-3', '3b9'),
                exact: true
              },
              {
                path: '/ur/docs/module-4/ch4-4',
                component: ComponentCreator('/ur/docs/module-4/ch4-4', '4c9'),
                exact: true
              },
              {
                path: '/ur/docs/part1-foundations/chapter-1-intro',
                component: ComponentCreator('/ur/docs/part1-foundations/chapter-1-intro', '52e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ur/docs/part1-foundations/chapter-2-ros2',
                component: ComponentCreator('/ur/docs/part1-foundations/chapter-2-ros2', '3c4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ur/docs/part2-simulation/chapter-3-urdf',
                component: ComponentCreator('/ur/docs/part2-simulation/chapter-3-urdf', '399'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ur/docs/part2-simulation/chapter-4-gazebo',
                component: ComponentCreator('/ur/docs/part2-simulation/chapter-4-gazebo', '8c7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ur/docs/part2-simulation/chapter-5-isaac',
                component: ComponentCreator('/ur/docs/part2-simulation/chapter-5-isaac', 'fa6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ur/docs/part3-perception/chapter-6-vision',
                component: ComponentCreator('/ur/docs/part3-perception/chapter-6-vision', '771'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ur/docs/part3-perception/chapter-7-slam',
                component: ComponentCreator('/ur/docs/part3-perception/chapter-7-slam', '157'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ur/docs/part4-ai-integration/chapter-8-vla',
                component: ComponentCreator('/ur/docs/part4-ai-integration/chapter-8-vla', 'cc2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ur/docs/part4-ai-integration/chapter-9-rl',
                component: ComponentCreator('/ur/docs/part4-ai-integration/chapter-9-rl', '8d6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ur/docs/part5-humanoid-dev/chapter-10-kinematics',
                component: ComponentCreator('/ur/docs/part5-humanoid-dev/chapter-10-kinematics', 'c91'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ur/docs/part5-humanoid-dev/chapter-11-hri',
                component: ComponentCreator('/ur/docs/part5-humanoid-dev/chapter-11-hri', 'ea9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ur/docs/part5-humanoid-dev/chapter-12-capstone',
                component: ComponentCreator('/ur/docs/part5-humanoid-dev/chapter-12-capstone', '330'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ur/docs/toc',
                component: ComponentCreator('/ur/docs/toc', '534'),
                exact: true
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/ur/',
    component: ComponentCreator('/ur/', '3b1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];

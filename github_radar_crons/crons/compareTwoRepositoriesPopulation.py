from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from api.ollama import chat_multi
import json

model = SentenceTransformer('all-MiniLM-L6-v2')


# Function to compute similarity score
def similarity_score(a, b):
    vec_a = model.encode([a], convert_to_tensor=True)
    vec_b = model.encode([b], convert_to_tensor=True)
    vec_a = vec_a.cpu().numpy().reshape(1, -1)  # Reshape to 2D array (1, n_features)
    vec_b = vec_b.cpu().numpy().reshape(1, -1)  # Reshape to 2D array (1, n_features)
    
    return cosine_similarity(vec_a, vec_b)[0][0]

# Function to summarize a project's README content
def get_project_summary(description):
    instruction_text = "Summarize the project's main idea and the problem it solves."
    instruction_text = """
--- TASK OVERVIEW ---

### INPUT FORMAT:
- You will be provided with the text of a github readme project.

### YOUR TASK:
Summarize the readme to get:
1) The project main idea 
2) What is the problem the project solves. 

### OUTPUT FORMAT:
- Return a **JSON object only** with the following structure:
{   
    "main_idea": "[The project main idea]",  
    "solving": "[What is the problem the project solves]"
}


### IMPORTANT NOTES:
- 📜 **Respond with valid JSON only**: No additional text, introductions, or summaries.
"""
    formatted_input = description
    seeded_prompt = "{" # You can add any pre-seeding context here if needed
    
    response = chat_multi([
        {"role": "system", "content": instruction_text},
        {"role": "user", "content": formatted_input},
        {"role": "assistant", "content": seeded_prompt}
    ])
    response_content = response['message']['content'].strip()

    # Ensure response starts with '[' and ends with ']' (fallback if needed)
    if not response_content.startswith("{"):
        response_content = "{" + response_content
    if not response_content.endswith("}"):
        response_content += "}"
    print(response_content)
    return json.loads(response_content)

# Compare two repositories: React vs Vue
def compareTwoRepositoriesPopulation(): 
    # List of React projects
    react_projects = [
        {
            "name": "tanstack/query", 
            "owner": "tanstack", 
            "readme": "TanStack Query Hooks for fetching, caching and updating asynchronous data in React, Solid, Svelte and Vue. Enjoy this library? Try the entire TanStack! TanStack Table, TanStack Router, TanStack Virtual, React Charts, React Ranger. Visit tanstack.com/query for docs, guides, API and more! Quick Features: Transport/protocol/backend agnostic data fetching (REST, GraphQL, promises, whatever!), Auto Caching + Refetching, Parallel + Dependent Queries, Mutations + Reactive Query Refetching, Multi-layer Cache + Automatic Garbage Collection, Paginated + Cursor-based Queries, Load-More + Infinite Scroll Queries, Request Cancellation, React Suspense + Fetch-As-You-Render Query Prefetching, Dedicated Devtools. Contributing View the contributing guidelines here."
        },
        {
            "name": "shadcn/ui", 
            "owner": "shadcn", 
            "readme": "shadcn/ui Accessible and customizable components that you can copy and paste into your apps. Free. Open Source. Use this to build your own component library. Documentation Visit http://ui.shadcn.com/docs to view the documentation. Contributing Please read the contributing guide. License Licensed under the MIT license. Hooks for fetching, caching and updating asynchronous data in React, Solid, Svelte and Vue Visit tanstack.com/query for docs, guides, API and more! Quick Features: Transport/protocol/backend agnostic data fetching (REST, GraphQL, promises, whatever!). Auto Caching + Refetching (stale-while-revalidate, Window Refocus, Polling/Realtime). Parallel + Dependent Queries. Mutations + Reactive Query Refetching. Multi-layer Cache + Automatic Garbage Collection. Paginated + Cursor-based Queries. Load-More + Infinite Scroll Queries w/ Scroll Recovery. Request Cancellation. React Suspense + Fetch-As-You-Render Query Prefetching. Dedicated Devtools. Contributing View the contributing guidelines here Become a Sponsor!"
        },
        {
            "name": "react-router", 
            "owner": "remix-run", 
            "readme": "React Router is a multi-strategy router for React bridging the gap from React 18 to React 19. You can use it maximally as a React framework or minimally as a library with your own architecture. Getting Started - Framework, Getting Started - Library, Upgrade from v6, Upgrade from Remix, Changelog. Packages: react-router, @react-router/dev, @react-router/node, @react-router/cloudflare, @react-router/serve, @react-router/fs-routes. Previous Versions: v6, v5."
        },
        {
            "name": "redux", 
            "owner": "reduxjs", 
            "readme": "Redux is a JS library for predictable and maintainable global state management. It helps you write applications that behave consistently, run in different environments (client, server, and native), and are easy to test. On top of that, it provides a great developer experience, such as live code editing combined with a time traveling debugger. You can use Redux together with React, or with any other view library. The Redux core is tiny (2kB, including dependencies), and has a rich ecosystem of addons. Redux Toolkit is our official recommended approach for writing Redux logic. It wraps around the Redux core, and contains packages and functions that we think are essential for building a Redux app. Redux Toolkit builds in our suggested best practices, simplifies most Redux tasks, prevents common mistakes, and makes it easier to write Redux applications. Installation: The recommended way to start new apps with React and Redux Toolkit is by using our official Redux Toolkit + TS template for Vite, or by creating a new Next.js project using Next's with-redux template. Both of these already have Redux Toolkit and React-Redux configured appropriately for that build tool, and come with a small example app that demonstrates how to use several of Redux Toolkit's features. We do not currently have official React Native templates, but recommend these templates for standard React Native and for Expo: https://github.com/rahsheen/react-native-template-redux-typescript https://github.com/rahsheen/expo-template-redux-typescript npm install @reduxjs/toolkit react-redux For the Redux core library by itself: npm install redux For more details, see the Installation docs page. Documentation: The Redux core docs are located at https://redux.js.org, and include the full Redux tutorials, as well usage guides on general Redux patterns: Introduction, Tutorials, Usage Guides, FAQ, API Reference. The Redux Toolkit docs are available at https://redux-toolkit.js.org, including API references and usage guides for all of the APIs included in Redux Toolkit. Learn Redux: Redux Essentials Tutorial The Redux Essentials tutorial is a 'top-down' tutorial that teaches 'how to use Redux the right way', using our latest recommended APIs and best practices. We recommend starting there. Redux Fundamentals Tutorial The Redux Fundamentals tutorial is a 'bottom-up' tutorial that teaches 'how Redux works' from first principles and without any abstractions, and why standard Redux usage patterns exist. Help and Discussion: The #redux channel of the Reactiflux Discord community is our official resource for all questions related to learning and using Redux. Reactiflux is a great place to hang out, ask questions, and learn - please come and join us there! Before Proceeding Further: Redux is a valuable tool for organizing your state, but you should also consider whether it's appropriate for your situation. Please don't use Redux just because someone said you should - instead, please take some time to understand the potential benefits and tradeoffs of using it. Here are some suggestions on when it makes sense to use Redux: You have reasonable amounts of data changing over time You need a single source of truth for your state You find that keeping all your state in a top-level component is no longer sufficient Yes, these guidelines are subjective and vague, but this is for a good reason. The point at which you should integrate Redux into your application is different for every user and different for every application. For more thoughts on how Redux is meant to be used, please see: When (and when not) to reach for Redux You Might Not Need Redux The Tao of Redux, Part 1 - Implementation and Intent The Tao of Redux, Part 2 - Practice and Philosophy Redux FAQ Basic Example: The whole global state of your app is stored in an object tree inside a single store. The only way to change the state tree is to create an action, an object describing what happened, and dispatch it to the store. To specify how state gets updated in response to an action, you write pure reducer functions that calculate a new state based on the old state and the action. Redux Toolkit simplifies the process of writing Redux logic and setting up the store. With Redux Toolkit, the basic app logic looks like: import { createSlice, configureStore } from '@reduxjs/toolkit' const counterSlice = createSlice({ name: 'counter', initialState: { value: 0 }, reducers: { incremented: state => { state.value += 1 }, decremented: state => { state.value -= 1 } } }) export const { incremented, decremented } = counterSlice.actions const store = configureStore({ reducer: counterSlice.reducer }) store.subscribe(() => console.log(store.getState())) store.dispatch(incremented()) store.dispatch(incremented()) store.dispatch(decremented()) Redux Toolkit allows us to write shorter logic that's easier to read, while still following the original core Redux behavior and data flow. Logo: You can find the official logo on GitHub. Change Log: This project adheres to Semantic Versioning. Every release, along with the migration instructions, is documented on the GitHub Releases page. License: MIT."
        },
        {
            "name": "react-bootstrap", 
            "owner": "react-bootstrap", 
            "readme": "React-Bootstrap is a library that provides Bootstrap 5 components built with React. It supports various Bootstrap versions, including v5.x with React-Bootstrap v2.x, v4.x with React-Bootstrap v1.x (not maintained), and v3.x with React-Bootstrap v0.33.x (not maintained). For migration from Bootstrap 4 to Bootstrap 5, refer to the documentation on migrating to React-Bootstrap v2, and for migration from Bootstrap 3 to 4, check out the v1 migration docs. It also integrates with React Router via the react-router-bootstrap module and offers additional components via Awesome React Bootstrap Components. For local setup, use Yarn, run yarn run bootstrap to install dependencies, and for testing, use yarn test or yarn run tdd for watch mode. To view a local copy of the documentation, use yarn start, or build the library with yarn run build. For CodeSandbox examples, explore the CodeSandbox repository. Contributions are welcome via the contributing guidelines."
        },
        {
            "name": "react-query", 
            "owner": "tanstack", 
            "readme": "TanStack Query provides hooks for fetching, caching, and updating asynchronous data in React, Solid, Svelte, and Vue. It supports various backends and protocols like REST, GraphQL, promises, etc. The library offers auto caching and refetching (stale-while-revalidate, Window Refocus, Polling/Realtime), parallel and dependent queries, mutations, multi-layer cache with automatic garbage collection, paginated and cursor-based queries, load-more and infinite scroll with scroll recovery, request cancellation, React Suspense for fetch-as-you-render, and dedicated devtools. For more, check out the documentation for different versions (v2, v3, v4), contribute by following the contributing guidelines, or become a sponsor to support development."
        },
        {
            "name": "react-hook-form", 
            "owner": "react-hook-form", 
            "readme": "React Hook Form is a simple React hook for form validation with performance, UX, and DX in mind, integrating seamlessly with UI libraries and offering features like built-in HTML validation, a small size, and support for libraries like Yup, Zod, AJV, Superstruct, Joi, and more. To install, use npm install react-hook-form, and to get started, refer to the API and form builder resources. It also offers examples, FAQs, and community support via Discord. Several sponsors and contributors help maintain the project, which is also backed by Vercel."
        },
        {
            "name": "styled-components", 
            "owner": "styled-components", 
            "readme": "Styled-components is a library that lets you use ES6 and CSS to style your React apps without hassle. It allows you to write real CSS code using tagged template literals or style objects. Styled-components are compatible with both React and React Native, perfect for building universal apps. The library provides features like theming, server-side rendering, and modular formats such as UMD, CJS, and ESM. For installation, refer to the docs. Contributions, backers, and sponsors are welcomed, and the project is open-sourced under the MIT License."
        },
    ]

    # List of Vue projects
    vue_projects = [
        {
            "name": "nuxt/nuxt", 
            "owner": "nuxt", 
            "readme": "Nuxt is a free and open-source framework with an intuitive and extendable way to create type-safe, performant and production-grade full-stack web applications and websites with Vue.js. It provides a number of features that make it easy to build fast, SEO-friendly, and scalable web applications, including: - Server-side rendering, Static Site Generation, Hybrid Rendering and Edge-Side Rendering - Automatic routing with code-splitting and pre-fetching - Data fetching and state management - SEO Optimization and Meta tags definition - Auto imports of components, composables and utils - TypeScript with zero configuration - Go fullstack with our server/ directory - Extensible with 200+ modules - Deployment to a variety of hosting platforms - ...and much more 🚀"
        },
        {
            "name": "element-plus", 
            "owner": "element-plus", 
            "readme": "Element Plus - A Vue.js 3 UI library - 💪 Vue 3 Composition API - 🔥 Written in TypeScript Getting Started Alright, for you to get started if you are looking for making Element Plus better you should keep reading. For developers that uses Element Plus to develop your website you should go ahead visit Getting Started. - 中国大陆加速镜像站点 Breaking change list The first stable release of Element Plus suitable for use in production was released on February 07, 2022. The APIs is stable right now, and here's also a full list about how to get upgraded from Element UI to Element Plus. You can find the breaking change list here: Breaking Change List. Migration Tool We have made a migration tool for you to migrate your project from Element UI to Element Plus. You can find the gogo code migration tool here. We have tested this on Vue Element Admin. You can find the transpiled code here. Playground You can also try Element Plus out with the components built-in playground. Try it with our built-in playground Playground Try it with code sandbox. Special thanks to the generous sponsorship by: Platinum Sponsors Gold Sponsors Translations Element Plus is translated to multiple languages, you can click the badge to help up update the translation or apply to become a proofreader. For now we are only showing English and Chinese for resource reasons, but we are looking forward to translate it into more languages, please go to the link above and leave a message if you want to help translating Element Plus into your desired language. How to help translating See how to help translating in Translating Element Plus. Stay tuned Join our Discord to start communicating with everybody. This thing is broken, I should help improve it! Awesommmmmmee. Everything you need is down below. You can also refer to CONTRIBUTING and Code of Conduct where you'll find the same information listed below. I would like to become a part of the development team! Welcome! We are looking for talented developers to join us and making Element Plus better! If you care to join the development team, please reach out to us, you are more than welcomed to join us! We are now lacking of experts of Testing, GitHub Actions, PM, if you do feel like you can and willing to help us, please do reach out to us. Contributors This project exists thanks to all the people who contribute. And thank you to all our backers! License Element Plus is open source software licensed as MIT."
        },
        {
            "name": "vue-router", 
            "owner": "vuejs", 
            "readme": "Compatibility Note: vue-router requires Vue.js 0.12.10+ and currently does not support Vue.js 2.0. ### Introduction vue-router is the official router for Vue.js. It deeply integrates with Vue.js core to make building Single Page Applications with Vue.js a breeze. Features include: - Nested route/view mapping - Modular, component-based router configuration - Route params, query, wildcards - View transition effects powered by Vue.js' transition system - Fine-grained navigation control - Links with automatic active CSS classes - HTML5 history mode or hash mode, with auto-fallback in IE9 - Restore scroll position when going back in history mode Get started with the documentation. ### Development Setup bash # install deps npm install # build dist files npm run build # serve example app at localhost:8080 npm run serve-example # and unit tests at localhost:8081 npm run dev # lint & run all tests npm test # run unit tests only npm run unit # run e2e tests only npm run e2e-local ## Questions For questions and support please use the Gitter chat room or the official forum. The issue list of this repo is exclusively for bug reports and feature requests. ## Issues Please make sure to read the Issue Reporting Checklist before opening an issue. Issues not conforming to the guidelines may be closed immediately. ## Contribution Please make sure to read the Contributing Guide before making a pull request. ## Changelog Details changes for each release are documented in the release notes. ## Stay In Touch - For latest releases and announcements, follow on Twitter: @vuejs ## License MIT Copyright (c) 2013-2016 Evan You"
        },
        {
            "name": "vuex", 
            "owner": "vuejs", 
            "readme": "--- Pinia is now the new default The official state management library for Vue has changed to Pinia. Pinia has almost the exact same or enhanced API as Vuex 5, described in Vuex 5 RFC. You could simply consider Pinia as Vuex 5 with a different name. Pinia also works with Vue 2.x as well. Vuex 3 and 4 will still be maintained. However, it's unlikely to add new functionalities to it. Vuex and Pinia can be installed in the same project. If you're migrating existing Vuex app to Pinia, it might be a suitable option. However, if you're planning to start a new project, we highly recommend using Pinia instead. --- Vuex is a state management pattern + library for Vue.js applications. It serves as a centralized store for all the components in an application, with rules ensuring that the state can only be mutated in a predictable fashion. It also integrates with Vue's official devtools extension to provide advanced features such as zero-config time-travel debugging and state snapshot export / import. Learn more about Vuex at 'What is Vuex', or get started by looking into full documentation. ## Documentation To check out docs, visit vuex.vuejs.org. ## Examples You may find example applications built with Vuex under the examples directory. Running the examples: bash $ npm install $ npm run dev # serve examples at localhost:8080 ## Questions For questions and support please use the Discord chat server or the official forum. The issue list of this repo is exclusively for bug reports and feature requests. ## Issues Please make sure to read the Issue Reporting Checklist before opening an issue. Issues not conforming to the guidelines may be closed immediately. ## Changelog Detailed changes for each release are documented in the release notes. ## Stay In Touch For latest releases and announcements, follow on Twitter: @vuejs. ## Contribution Please make sure to read the Contributing Guide before making a pull request. ## License MIT Copyright (c) 2015-present Evan You"
        },
        {
            "name": "vuetify", 
            "owner": "vuetifyjs", 
            "readme": "Vuetify is a UI Library with beautifully handcrafted Vue Components. Vuetify is a project developed and maintained by John Leider and Heather Leider, with support from the entire Core Team. Vuetify provides various ways to support the development of the project, including GitHub Sponsors, Patreon, OpenCollective, Tidelift, and one-time donations through PayPal. Users can also book time with the team for support.Vuetify has a powerful color system for styling applications, a responsive layout, long-term support for major releases, and supports 42+ languages for internationalization. The framework is compatible with modern browsers and includes SASS/SCSS customization options and a theme system.In addition, Vuetify provides useful resources, including Vuetify Snips for pre-built code snippets, enterprise support, and a community Discord server. Users can also explore Vuetify Play, a playground for testing components, and check for issues on Vuetify Issues. For those looking for pre-made themes or digital products, Vuetify's official store is also available.The documentation for Vuetify is available on vuetifyjs.com, and contributors are encouraged to check the Code of Conduct and Contribution Guide before contributing. Vuetify is licensed under the MIT License, and users can view the full list of contributors and get involved in the community."
        },
        {
            "name": "quasar", 
            "owner": "quasarframework", 
            "readme": "Build high-performance VueJS user interfaces in record time: responsive Single Page Apps, SSR Apps, PWAs, Browser extensions, Hybrid Mobile Apps, and Electron Apps. If you want, all using the same codebase!Quasar Framework is an MIT-licensed open source project. Its ongoing development is made possible thanks to the support by these awesome backers. Please read our manifest on Why donations are important. If you'd like to become a donator, check out Quasar Framework's Donator campaign.Supporting Quasar Proudly sponsored by: - Dreamonkey - Hapag Lloyd - iRewind - Qintil - Synestia - Letsbutterfly - Kalisio - COMCOMService - DigitalOcean - ProjectFinance - iBL - JuggleStreet - Nodesol - AcornFinance - Certible - BairesDevDocumentation Head on to the Quasar Framework official website: https://quasar.devStay in Touch For the latest releases and announcements, follow us on our Twitter account: @quasarframeworkChat Support Ask questions at the official community Discord server: https://chat.quasar.devCommunity Forum Ask questions at the official community forum: https://forum.quasar.devContributing Please make sure to read the Contributing Guide before making a pull request. If you have a Quasar-related project/component/tool, add it with a pull request to this curated list!Thank you to all the people who have already contributed to Quasar!Semver Quasar is following Semantic Versioning 2.0.License Copyright (c) 2015-present Razvan Stoenescu MIT License"
        },
        {
            "name": "vite", 
            "owner": "vitejs", 
            "readme": "Vite ⚡\n\nNext Generation Frontend Tooling\n\n- 💡 Instant Server Start\n- ⚡️ Lightning Fast HMR\n- 🛠️ Rich Features\n- 📦 Optimized Build\n- 🔩 Universal Plugin Interface\n- 🔑 Fully Typed APIs\n\nVite (French word for \"quick\", pronounced /vit/, like \"veet\") is a new breed of frontend build tooling that significantly improves the frontend development experience. It consists of two major parts:\n\n- A dev server that serves your source files over native ES modules, with rich built-in features and astonishingly fast Hot Module Replacement (HMR).\n\n- A build command that bundles your code with Rollup, pre-configured to output highly optimized static assets for production.\n\nVite is highly extensible via its Plugin API and JavaScript API with full typing support.\n\nRead the Docs to Learn More: https://vite.dev\n\nPackages:\n- vite: [https://img.shields.io/npm/v/vite.svg?label=%20](packages/vite/CHANGELOG.md)\n- @vitejs/plugin-legacy: [https://img.shields.io/npm/v/@vitejs/plugin-legacy.svg?label=%20](packages/plugin-legacy/CHANGELOG.md)\n- create-vite: [https://img.shields.io/npm/v/create-vite.svg?label=%20](packages/create-vite/CHANGELOG.md)\n\nContribution: See Contributing Guide (CONTRIBUTING.md).\nLicense: MIT\nSponsors: [https://github.com/sponsors/yyx990803](https://github.com/sponsors/yyx990803)\n![sponsors](https://sponsors.vuejs.org/vite.svg?v2)"
        },
        {
            "name": "vue-i18n", 
            "owner": "kazupon", 
            "readme": "ue-i18n\nBuild Status\nCoverage Status\nNPM version\nvue-i18n channel on Discord\nvue-i18n Dev Token\nInternationalization plugin for Vue.js\nSilver Sponsors\nDetailed changes for each release are documented in the CHANGELOG.md.\nIssues\nPlease make sure to read the Issue Reporting Checklist before opening an issue. Issues not conforming to the guidelines may be closed immediately.\nContribution\nPlease make sure to read the Contributing Guide before making a pull request."
        },
        {
            "name": "vue-cli", 
            "owner": "vuejs", 
            "readme": "Vue CLI\nBuild Status\nWindows Build status\nlerna\nVue CLI is now in maintenance mode. For new Vue 3 projects, please use create-vue to scaffold Vite-based projects.\nDocumentation\nDocs are available at https://cli.vuejs.org/ - we are still working on refining it and contributions are welcome!\nContributing\nPlease see contributing guide.\nLicense\nMIT"
        },
        {
            "name": "pinia", 
            "owner": "vuejs", 
            "readme": "Pinia\n\nIntuitive, type safe and flexible Store for Vue\n\n- 💡 Intuitive\n- 🔑 Type Safe\n- ⚙️ Devtools support\n- 🔌 Extensible\n- 🏗 Modular by design\n- 📦 Extremely light\n- ⛰️ Nuxt Module\n\nThe latest version of pinia works with Vue 3. See the branch v2 for a version that works with Vue 2.\n\nPinia is the most similar English pronunciation of the word _pineapple_ in Spanish: _piña_. A pineapple is in reality a group of individual flowers that join together to create a multiple fruit. Similar to stores, each one is born individually, but they are all connected at the end. It's also a delicious tropical fruit indigenous to South America.\n\n👉 Demo with Vue 3 on StackBlitz: https://stackblitz.com/github/piniajs/example-vue-3-vite\n👉 Demo with Nuxt 3 on StackBlitz: https://stackblitz.com/github/piniajs/example-nuxt-3\n\nHelp me keep working on this project 💚\n- Become a Sponsor on GitHub: https://github.com/sponsors/posva\n- One-time donation via PayPal: https://paypal.me/posva\n\nSilver Sponsors\n- [Route Optimizer and Route Planner Software](https://route4me.com)\n- [Prefect](https://www.prefect.io/)\n- [VueMastery](https://www.vuemastery.com/)\n\nBronze Sponsors\n- [Storyblok](https://storyblok.com)\n- [Nuxt UI Pro Templates](https://ui.nuxt.com/pro)\n- [Antony Konstantinidis](https://www.vuejs.de)\n- [Stanislas Ormières](https://stormier.ninja)\n\nFAQ\n\nQ: _Is Pinia the successor of Vuex?_ A: Yes\nQ: _What about dynamic modules?_ A: Dynamic modules are not type safe, so instead we allow creating different stores that can be imported anywhere.\n\nInstallation\n```\n# or pnpm or yarn\nnpm install pinia\n```\nUsage\nCreate a pinia (the root store) and pass it to app:\n```js\nimport { createApp } from 'vue'\nimport { createPinia } from 'pinia'\nimport App from './App.vue'\n\nconst pinia = createPinia()\nconst app = createApp(App)\n\napp.use(pinia)\napp.mount('#app')\n```\nFor more detailed instructions, including Nuxt configuration, check the Documentation: https://pinia.vuejs.org.\n\nCreate a Store\n```ts\nimport { defineStore } from 'pinia'\n\n// main is the name of the store. It is unique across your application\n// and will appear in devtools\nexport const useMainStore = defineStore('main', {\n  state: () => ({\n    counter: 0,\n    name: 'Eduardo',\n  }),\n  getters: {\n    doubleCounter: (state) => state.counter * 2,\n    doubleCounterPlusOne(): number {\n      return this.doubleCounter + 1\n    },\n  },\n  actions: {\n    reset() {\n      this.counter = 0\n    },\n  },\n})\n```\n`defineStore` returns a function that has to be called to get access to the store:\n```ts\nimport { useMainStore } from '@/stores/main'\nimport { storeToRefs } from 'pinia'\n\nexport default defineComponent({\n  setup() {\n    const main = useMainStore()\n    const { counter, doubleCounter } = storeToRefs(main)\n\n    return {\n      main,\n      counter,\n      doubleCounter,\n    }\n  },\n})\n```\nDocumentation: https://pinia.vuejs.org\nLicense: MIT"
        }
    ]

    # Fetch and summarize README content for React projects
    react_summaries = {}
    for project in react_projects:
        content = get_project_summary(project['readme'])
        print(content)
        react_summaries[project['name']] = content['main_idea'] + ' ' +  content['solving'] # Summarize the README

    # Fetch and summarize README content for Vue projects
    vue_summaries = {}
    for project in vue_projects:
        content = get_project_summary(project['readme'])
        vue_summaries[project['name']] = content['main_idea'] + ' ' +  content['solving'] # Summarize the README

    # Print out the React projects that have no highly similar Vue counterparts
    print("\nReact projects with no highly similar Vue counterparts:")
    print(react_summaries)
    for r_name, r_summary in react_summaries.items():
        scores = [(v_name, similarity_score(r_summary, v_summary)) for v_name, v_summary in vue_summaries.items()]
        if scores:
            closest = max(scores, key=lambda x: x[1])
            print(f"- {r_name} (Closest Vue project: {closest[0]}, Similarity: {closest[1]:.2f})")
        else:
            print(f"- {r_name} (No Vue projects available for comparison)")

# Run the comparison function
compareTwoRepositoriesPopulation()

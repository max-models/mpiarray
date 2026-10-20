// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/mpiarray/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'mpiarray',
	title: 'mpiarray',
	description: 'MPI domain decomposition and distributed NumPy/CuPy arrays with halo exchange.',
	pythonPackage: 'mpiarray',
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/mpiarray/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
			],
		},
		{
			label: 'Guides',
			items: [{ autogenerate: { directory: 'guides' } }],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
			],
		},
	],
	astro: {
		redirects: {
			'/tutorials/': '/guides/',
			'/tutorials/distributed-arrays/': '/guides/distributed-arrays/',
			'/tutorials/heat-equation/': '/guides/heat-equation/',
			'/tutorials/particles/': '/guides/particles/',
			'/tutorials/working-with-data/': '/guides/working-with-data/',
			'/tutorials/units/': '/guides/units/',
		},
	},
});

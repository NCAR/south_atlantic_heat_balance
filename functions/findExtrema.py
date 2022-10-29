def findExtrema(y, z, psi, min_lat=-90., max_lat=90., min_depth=0., mult=1., plot = True):
psiMax = mult*np.amax( mult * np.ma.array(psi)[(y>=min_lat) & (y<=max_lat) & (z<-min_depth)] )
idx = np.argmin(np.abs(psi-psiMax))
(j,i) = np.unravel_index(idx, psi.shape)
if plot:
	#plt.plot(y[j,i],z[j,i],'kx',hold=True)
	plt.plot(y[j,i],z[j,i],'kx')
    	plt.text(y[j,i],z[j,i],'%.1f'%(psi[j,i]),color='red', fontsize=12)
else:
return psi[j,i]

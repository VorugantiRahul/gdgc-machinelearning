import pandas as pd
import numpy as np

def extract_features(df):
    d = df.copy()
    
    # 1. Domain Invariant Attack Signatures
    d['sig_ftp_dos'] = ((d['fwd_init_window_size'] == 64.0) & (d['fwd_pkts_payload.avg'] == 120.0)).astype(float)
    d['sig_recon_scan'] = ((d['fwd_pkts_tot'] <= 2) & (d['fwd_pkts_payload.tot'] == 0) & (d['bwd_pkts_payload.tot'] == 0)).astype(float)
    d['sig_mqtt_benign'] = (d['service'] == 'mqtt').astype(float)
    
    # 2. Port Semantics & Groupings
    d['resp_is_well_known'] = (d['id.resp_p'] < 1024).astype(float)
    d['resp_is_registered'] = ((d['id.resp_p'] >= 1024) & (d['id.resp_p'] < 49152)).astype(float)
    d['resp_is_ephemeral'] = (d['id.resp_p'] >= 49152).astype(float)
    d['orig_is_well_known'] = (d['id.orig_p'] < 1024).astype(float)
    d['orig_is_ephemeral'] = (d['id.orig_p'] >= 49152).astype(float)
    d['same_orig_resp_port'] = (d['id.orig_p'] == d['id.resp_p']).astype(float)
    
    # Standard Protocol Ports
    d['port_ftp'] = (d['id.resp_p'] == 21.0).astype(float)
    d['port_ssh'] = (d['id.resp_p'] == 22.0).astype(float)
    d['port_dns'] = (d['id.resp_p'] == 53.0).astype(float)
    d['port_dhcp'] = ((d['id.resp_p'] == 67.0) | (d['id.resp_p'] == 68.0)).astype(float)
    d['port_http'] = (d['id.resp_p'] == 80.0).astype(float)
    d['port_ntp'] = (d['id.resp_p'] == 123.0).astype(float)
    d['port_netbios'] = (d['id.resp_p'] == 137.0).astype(float)
    d['port_https'] = (d['id.resp_p'] == 443.0).astype(float)
    d['port_mqtt'] = (d['id.resp_p'] == 1883.0).astype(float)
    d['port_mdns'] = (d['id.resp_p'] == 5353.0).astype(float)
    
    # 3. Flow Directionality & Ratio Metrics
    d['bwd_to_fwd_pkts'] = d['bwd_pkts_tot'] / np.maximum(d['fwd_pkts_tot'], 1.0)
    d['bwd_to_fwd_bytes'] = d['bwd_pkts_payload.tot'] / np.maximum(d['fwd_pkts_payload.tot'], 1.0)
    d['tot_pkts'] = d['fwd_pkts_tot'] + d['bwd_pkts_tot']
    d['tot_bytes'] = d['flow_pkts_payload.tot']
    
    d['fwd_pkt_mean_size'] = d['fwd_pkts_payload.tot'] / np.maximum(d['fwd_pkts_tot'], 1.0)
    d['bwd_pkt_mean_size'] = d['bwd_pkts_payload.tot'] / np.maximum(d['bwd_pkts_tot'], 1.0)
    d['tot_pkt_mean_size'] = d['tot_bytes'] / np.maximum(d['tot_pkts'], 1.0)
    
    d['is_unidirectional'] = (d['bwd_pkts_tot'] == 0).astype(float)
    d['is_zero_duration'] = (d['flow_duration'] == 0.0).astype(float)
    d['has_bwd_payload'] = (d['bwd_pkts_payload.tot'] > 0).astype(float)
    d['has_fwd_payload'] = (d['fwd_pkts_payload.tot'] > 0).astype(float)
    
    # 4. Header to Payload Ratios
    d['fwd_header_to_payload'] = d['fwd_header_size_tot'] / np.maximum(d['fwd_pkts_payload.tot'], 1.0)
    d['bwd_header_to_payload'] = d['bwd_header_size_tot'] / np.maximum(d['bwd_pkts_payload.tot'], 1.0)
    d['tot_header_size'] = d['fwd_header_size_tot'] + d['bwd_header_size_tot']
    
    # 5. TCP Flags Dynamics
    d['syn_only'] = ((d['flow_SYN_flag_count'] > 0) & (d['flow_ACK_flag_count'] == 0)).astype(float)
    d['fin_only'] = ((d['flow_FIN_flag_count'] > 0) & (d['flow_SYN_flag_count'] == 0)).astype(float)
    d['rst_present'] = (d['flow_RST_flag_count'] > 0).astype(float)
    d['psh_present'] = ((d['fwd_PSH_flag_count'] > 0) | (d['bwd_PSH_flag_count'] > 0)).astype(float)
    d['total_flag_count'] = (d['flow_FIN_flag_count'] + d['flow_SYN_flag_count'] + 
                             d['flow_RST_flag_count'] + d['fwd_PSH_flag_count'] + 
                             d['bwd_PSH_flag_count'] + d['flow_ACK_flag_count'] + 
                             d['flow_CWR_flag_count'] + d['flow_ECE_flag_count'])
    
    # 6. Window Size Fingerprints
    d['win_fwd_64'] = (d['fwd_init_window_size'] == 64.0).astype(float)
    d['win_fwd_0'] = (d['fwd_init_window_size'] == 0.0).astype(float)
    d['win_fwd_64240'] = (d['fwd_init_window_size'] == 64240.0).astype(float)
    d['win_fwd_65476'] = (d['fwd_init_window_size'] == 65476.0).astype(float)
    d['win_fwd_65535'] = (d['fwd_init_window_size'] == 65535.0).astype(float)
    d['win_bwd_0'] = (d['bwd_init_window_size'] == 0.0).astype(float)
    d['win_bwd_65160'] = (d['bwd_init_window_size'] == 65160.0).astype(float)
    d['win_bwd_26847'] = (d['bwd_init_window_size'] == 26847.0).astype(float)
    d['win_bwd_65535'] = (d['bwd_init_window_size'] == 65535.0).astype(float)
    
    # 7. Robust Rate and Duration Features
    eps = 1e-6
    d['log_duration'] = np.log1p(d['flow_duration'])
    d['robust_flow_pkts_per_sec'] = np.log1p((d['tot_pkts']) / np.maximum(d['flow_duration'], eps))
    d['robust_flow_bytes_per_sec'] = np.log1p(d['tot_bytes'] / np.maximum(d['flow_duration'], eps))
    d['robust_fwd_pkts_per_sec'] = np.log1p(d['fwd_pkts_tot'] / np.maximum(d['flow_duration'], eps))
    d['robust_bwd_pkts_per_sec'] = np.log1p(d['bwd_pkts_tot'] / np.maximum(d['flow_duration'], eps))
    
    # 8. Categorical Encodings (One-Hot Encoded)
    for p in ['tcp', 'udp', 'icmp']:
        d[f'proto_{p}'] = (d['proto'] == p).astype(float)
        
    services = ['-', 'dns', 'mqtt', 'http', 'ssl', 'ntp', 'dhcp', 'irc', 'radius']
    for s in services:
        d[f'service_{s}'] = (d['service'] == s).astype(float)
        
    # Drop raw string categoricals and id
    cols_to_drop = ['proto', 'service', 'fwd_URG_flag_count']
    d = d.drop(columns=[c for c in cols_to_drop if c in d.columns])
    
    return d

if __name__ == '__main__':
    train = pd.read_csv("c:/Users/RAHUL/Documents/network-intrusion-detection2-ml-challenge/train.csv")
    test = pd.read_csv("c:/Users/RAHUL/Documents/network-intrusion-detection2-ml-challenge/test.csv")
    
    tr_f = extract_features(train)
    te_f = extract_features(test)
    
    print(f"Train features shape: {tr_f.shape}")
    print(f"Test features shape: {te_f.shape}")
    feature_cols = [c for c in tr_f.columns if c not in ['id', 'Target']]
    print(f"Total model features: {len(feature_cols)}")
    print("Sample feature names:", feature_cols[:20])

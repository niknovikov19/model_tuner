from model_tuner.sim_manager import (
    SimResultLocator
)

from model_tuner.data_proc import (
    NetSpikesParams,
    NetRatesParams,
    DataKeeper,
    SimResultParserNetPyNE,
    DataProcessor
)


def get_sim_rates(
        dk: DataKeeper,
        sim_res_locator: SimResultLocator,
        sim_label: str,
        spikes_calc_params: NetSpikesParams,
        rates_calc_params: NetRatesParams
        ) -> dict[str, float]:

    # Locate sim result by sim label
    sim_result_desc = sim_res_locator.locate_result(sim_label)
    
    # Initialize sim result parser
    res_parser = SimResultParserNetPyNE(dk=dk, result_desc=sim_result_desc)
    
    # Initialize data processor
    data_proc = DataProcessor(dk)
    
    # Extract spikes from the sim result and strore them into dk
    spikes_data_id = res_parser.extract_net_spikes(
        spikes_calc_params,
        data_name_out=f'net_spikes_{sim_label}'
    )
    
    # Load spikes from dk, calculate rates from them, save rates into dk
    rates_data_id = data_proc.calc_net_rates(
        spikes_data_id,
        rates_calc_params,
        data_name_out=f'net_rates_{sim_label}',
        recalc=False
    )
    
    # Load rates from dk
    rates = data_proc.load_data(rates_data_id)
    return rates.data

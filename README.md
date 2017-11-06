# Smart Campaign Management Project

## The concept
Smart campaign engine for alpha and risk management within a campaign. This will be framework independent codebase, for backtesting and implementation of campaign management strategies.

## Functionality
- Alpha management universal engine - alphas rebalancing based on performance, alpha shutting off due to significant losses. 
- Campaign risk management engine - keep campaign risks (drawdowns) in constrained range using campaign size management.
- Capital agnostic risk management - campaign engine will provide universal risk estimation in dollars, which can be exposed to each account by using account capital and individual risk constraints.

## Engine workflow
- Estimate each alpha base risk (for example based on equity ATR)
- Equalize each alpha risks by applying adjustment coefficients to alphas
- (optional) Manage each alpha based on algorithmic logic (for example, relative strength based management.) We can turn on/off alphas or slightly change alpha size based on some logic.
- Calculate campaign risk - adjust campaign size to fit total risk contraints. At this step the risk will be some constant, finally it will be re-calculated for each account in the live stage.
- (live) Every weekend run the Smart campaing engine for every attached campaign and re-calculate the logic, and base campaign weights and risk estimaiton, alphas weights will be racalculated too.
- (live) Every Monday all of the accounts will get new weights according capital amount and alpha/campaing risk estimation.